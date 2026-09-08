#!/usr/bin/env python3
"""Shared dev/test metrics and checkpoint-only BIO evaluation CLI."""
from __future__ import annotations

# Support both "python baseline/file.py" and "python -m baseline.file".
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "baseline"

import json
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple

import torch

from .common import (
    TYPES, CharBioDataset, normalize_type, normalize_row_for_eval,
    bio_to_spans, read_rows, save_json, parse_args, build_runtime, release_datasets,
    _dist_is_initialized, _dist_rank, _dist_world_size,
)
from .inference import merge_and_decode, _infer_pred_spans

CORE_METRIC_KEYS = (
    "PER_precision", "PER_recall", "PER_f1",
    "LOC_precision", "LOC_recall", "LOC_f1",
    "BOOK_precision", "BOOK_recall", "BOOK_f1",
    "macro_precision", "macro_recall", "macro_f1",
    "micro_precision", "micro_recall", "micro_f1", "overall_f1",
    "seen_precision", "seen_recall", "seen_f1",
    "unseen_precision", "unseen_recall", "unseen_f1",
    "seen_PER_f1", "seen_LOC_f1", "seen_BOOK_f1",
    "unseen_PER_f1", "unseen_LOC_f1", "unseen_BOOK_f1",
    "error_missing", "error_spurious", "error_boundary_error", "error_type_error",
)

def core_metrics(gold, pred, texts, seen):
    """Dev adapter: use the same paragraph-level counters as streamed test."""
    accumulator = MetricAccumulator(seen)
    for pid, spans in gold.items():
        accumulator.update_spans(spans, pred.get(pid, []), texts.get(pid, ""))
    return accumulator.results()

def _normalize_eval_spans(raw_spans: Any) -> List[Tuple[int, int, str]]:
    out: List[Tuple[int, int, str]] = []
    if raw_spans is None:
        return out

    try:
        iterator = iter(raw_spans)
    except TypeError:
        return out

    for span in iterator:
        if isinstance(span, dict):
            s = span.get("start")
            e = span.get("end")
            t = span.get("type")
        else:
            try:
                if len(span) < 3:
                    continue
                s, e, t = span[0], span[1], span[2]
            except (TypeError, IndexError, KeyError):
                continue

        try:
            s = int(s)
            e = int(e)
        except (TypeError, ValueError):
            continue

        t = normalize_type(str(t))
        if t not in TYPES or e <= s:
            continue
        out.append((s, e, t))
    return out

def _prf_counts(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f

def count_span_errors(
    gold: List[Tuple[int, int, str]],
    pred: List[Tuple[int, int, str]],
) -> Counter:
    """Match exact spans first, then greedily pair positive-overlap spans.

    Preserve gold/pred order: overlap ties prefer the same type, then the
    first prediction. These diagnostic counts do not change strict F1.
    """
    counts = Counter({"missing": 0, "spurious": 0, "boundary_error": 0, "type_error": 0})
    pred_by_key: Dict[Tuple[int, int, str], List[int]] = {}
    for idx, span in enumerate(pred):
        pred_by_key.setdefault(span, []).append(idx)
    used_pred = set()
    unmatched_gold = []
    for g in gold:
        match_idx = next((idx for idx in pred_by_key.get(g, [])
                          if idx not in used_pred), None)
        if match_idx is None:
            unmatched_gold.append(g)
        else:
            used_pred.add(match_idx)

    for g in unmatched_gold:
        gs, ge, gt = g
        best_idx = None
        best_overlap = 0
        best_same_type = False
        for idx, p in enumerate(pred):
            if idx in used_pred:
                continue
            ps, pe, pt = p
            overlap = max(0, min(ge, pe) - max(gs, ps))
            if overlap <= 0:
                continue
            same_type = gt == pt
            if overlap > best_overlap or (overlap == best_overlap and same_type and not best_same_type):
                best_idx = idx
                best_overlap = overlap
                best_same_type = same_type
        if best_idx is None:
            counts["missing"] += 1
            continue
        used_pred.add(best_idx)
        if pred[best_idx][2] == gt:
            counts["boundary_error"] += 1
        else:
            counts["type_error"] += 1

    counts["spurious"] = len(pred) - len(used_pred)
    return counts

class MetricAccumulator:
    """Accumulate exact integer counts without retaining the full test corpus."""
    def __init__(self, seen):
        self.seen = seen
        self.counts = Counter()

    def update(self, rows, predictions):
        for i, row in enumerate(rows):
            pid, text, bio = normalize_row_for_eval(row, i)
            self.update_spans(bio_to_spans(bio), predictions.get(pid, []), text)

    def update_spans(self, gold, pred, text):
        """Count one complete paragraph; never average chunk/rank F1 values."""
        gold = _normalize_eval_spans(gold)
        pred = _normalize_eval_spans(pred)
        for name, count in count_span_errors(gold, pred).items():
            self.counts[("error", name)] += count
        gset, pset = set(gold), set(pred)
        for category, spans in (("TP", gset & pset), ("FN", gset - pset), ("FP", pset - gset)):
            for start, end, typ in spans:
                surface = text[max(0, start):min(len(text), end)]
                bucket = "seen" if surface in self.seen.get(typ, set()) else "unseen"
                self.counts[(bucket, typ, category)] += 1

    @staticmethod
    def counter_keys():
        """Fixed ordering shared by every distributed rank, including empty ranks."""
        return ([(b, t, c) for b in ("seen", "unseen") for t in TYPES
                 for c in ("TP", "FP", "FN")]
                + [("error", name) for name in
                   ("missing", "spurious", "boundary_error", "type_error")])

    def results(self):
        out = {}
        def add(prefix, buckets, types, f1_only=False):
            tp, fp, fn = [sum(self.counts[(b, t, c)] for b in buckets for t in types)
                          for c in ("TP", "FP", "FN")]
            p, r, f = _prf_counts(tp, fp, fn)
            out[prefix + "_f1"] = f
            if not f1_only:
                out.update({prefix + "_precision": p, prefix + "_recall": r})
        for typ in TYPES:
            add(typ, ("seen", "unseen"), (typ,))
        for metric in ("precision", "recall", "f1"):
            out["macro_" + metric] = sum(out[t + "_" + metric] for t in TYPES) / len(TYPES)
        add("micro", ("seen", "unseen"), TYPES)
        out["overall_f1"] = out["micro_f1"]
        for bucket in ("seen", "unseen"):
            add(bucket, (bucket,), TYPES)
            for typ in TYPES:
                add(bucket + "_" + typ, (bucket,), (typ,), f1_only=True)
        for name in ("missing", "spurious", "boundary_error", "type_error"):
            out["error_" + name] = int(self.counts[("error", name)])
        return {key: out[key] for key in CORE_METRIC_KEYS}

def evaluate_test(trainer, tokenizer, args, seen):
    from itertools import islice
    rank, world = _dist_rank(), _dist_world_size()
    shard = Path(args.output_dir) / f".test.rank{rank}.predictions.jsonl"
    accumulator = MetricAccumulator(seen)
    rows_iter = iter(read_rows(args.test_jsonl, args.limit_test_rows, rank, world))
    processed = 0
    with shard.open("w", encoding="utf-8") as output:
        while True:
            rows = list(islice(rows_iter, args.test_chunk_rows))
            if not rows:
                break
            ds = CharBioDataset(rows, tokenizer, args.max_len, args.stride)
            ds.distributed_prepartitioned = True
            bios, spans = _infer_pred_spans(
                trainer, ds, args.eval_batch_size, args.num_workers, not args.use_cpu)
            accumulator.update(rows, spans)
            for row in rows:
                pid = row["id"]
                output.write(json.dumps({
                    "id": pid, "text_len": len(row["text"]),
                    "pred_bio": bios.get(pid, ["O"] * len(row["text"])),
                    "pred_spans": spans.get(pid, []),
                }, ensure_ascii=False) + "\n")
            processed += len(rows)
            print(f"[test rank{rank}] {processed:,} paragraphs", flush=True)
            del ds, rows, bios, spans
    keys = accumulator.counter_keys()
    if _dist_is_initialized():
        counts = torch.tensor([accumulator.counts[k] for k in keys],
                              dtype=torch.long, device=trainer.args.device)
        torch.distributed.all_reduce(counts)
        accumulator.counts = Counter(dict(zip(keys, counts.cpu().tolist())))
        torch.distributed.barrier()
    if rank == 0:
        save_json(Path(args.output_dir) / "test_metrics.json", accumulator.results())
        # Restore input paragraph order from rank-strided shards, without collecting them in RAM.
        from contextlib import ExitStack
        with ExitStack() as stack:
            files = [stack.enter_context((Path(args.output_dir) /
                     f".test.rank{r}.predictions.jsonl").open(encoding="utf-8")) for r in range(world)]
            output = stack.enter_context((Path(args.output_dir) /
                     "test_predictions.jsonl").open("w", encoding="utf-8"))
            while True:
                lines = [handle.readline() for handle in files]
                if not any(lines):
                    break
                for line in lines:
                    output.write(line)
        print("[TEST]", json.dumps(accumulator.results()), flush=True)
    if _dist_is_initialized():
        torch.distributed.barrier()

def make_dev_metrics(dev_ds, dev_rows, seen):
    """Adapt Trainer's window predictions to the shared paragraph metrics."""
    def dev_metrics(eval_pred):
        predictions, _ = eval_pred
        pids = [w.pid for w in dev_ds.windows]
        maps = [torch.tensor(w.char_map) for w in dev_ds.windows]
        _, spans = merge_and_decode(dev_ds, predictions, pids, maps)
        gold, texts = {}, {}
        for i, row in enumerate(dev_rows):
            pid, text, bio = normalize_row_for_eval(row, i)
            gold[pid], texts[pid] = bio_to_spans(bio), text
        return core_metrics(gold, spans, texts, seen)

    return dev_metrics


def main():
    args = parse_args(evaluation_only=True)
    runtime = build_runtime(args)
    release_datasets(runtime)
    evaluate_test(runtime.trainer, runtime.tokenizer, args, runtime.seen)


if __name__ == "__main__":
    main()

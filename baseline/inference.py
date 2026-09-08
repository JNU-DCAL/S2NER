"""CRF window prediction, paragraph partitioning and character BIO merging."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForTokenClassification, Trainer

from .common import (
    CharBioDataset, LABELS, LABEL2ID, ID2LABEL, repair_bio, bio_to_spans,
    _dist_rank, _dist_world_size,
)

def merge_and_decode(eval_dataset: CharBioDataset,
                     logits: np.ndarray,
                     pids: List[str],
                     char_maps: List[torch.Tensor]) -> Tuple[Dict[str,List[str]], Dict[str,List[Tuple[int,int,str]]]]:
    char_len = eval_dataset.char_lengths
    agg_logits = {pid: np.zeros((char_len[pid], len(LABELS)), dtype=np.float64) for pid in char_len}
    agg_count  = {pid: np.zeros((char_len[pid],), dtype=np.int32) for pid in char_len}
    if logits.ndim != 2:
        raise ValueError("Expected decoded CRF label IDs with shape [windows, tokens]")
    for i in range(len(pids)):
        pid=pids[i]; cmap=char_maps[i].numpy(); lg=logits[i]
        for t_idx, ch_idx in enumerate(cmap):
            if ch_idx < 0 or ch_idx >= agg_logits[pid].shape[0]:
                continue
            lab = int(lg[t_idx])
            if lab < 0 or lab >= len(LABELS):
                continue
            agg_logits[pid][ch_idx, lab] += 1.0
            agg_count[pid][ch_idx] += 1
    pred_char_bio={}; pred_spans={}
    for pid in agg_logits:
        L=agg_logits[pid].shape[0]
        bio_ids=np.zeros(L, dtype=np.int32)
        for j in range(L):
            if agg_count[pid][j]==0: bio_ids[j]=LABEL2ID["O"]
            else: bio_ids[j]=int(np.argmax(agg_logits[pid][j]))
        bio_tags=[ID2LABEL[i] for i in bio_ids]
        bio_tags=repair_bio(bio_tags)
        pred_char_bio[pid]=bio_tags
        pred_spans[pid]=bio_to_spans(bio_tags)
    return pred_char_bio, pred_spans

def _local_pid_partition(dataset: CharBioDataset) -> Tuple[List[str], List[int]]:
    pid_order = list(dataset.char_lengths)

    if getattr(dataset, "distributed_prepartitioned", False):
        local_pids = pid_order
        local_indices = list(range(len(dataset)))
        return local_pids, local_indices

    world_size = _dist_world_size()
    rank = _dist_rank()
    local_pids = pid_order[rank::world_size] if world_size > 1 else pid_order
    local_pid_set = set(local_pids)
    local_indices = [idx for idx, window in enumerate(dataset.windows) if window.pid in local_pid_set]
    return local_pids, local_indices

def _prepare_model_for_eval(trainer: Trainer, dataloader: Optional[DataLoader] = None):
    model = trainer.model_wrapped if getattr(trainer, "model_wrapped", None) is not None else trainer.model

    model = trainer._wrap_model(model, training=False, dataloader=dataloader)

    target_dtype = None
    if not trainer.is_in_train:
        if trainer.args.bf16 or trainer.args.bf16_full_eval:
            target_dtype = torch.bfloat16
        elif trainer.args.fp16 or trainer.args.fp16_full_eval:
            target_dtype = torch.float16

    if target_dtype is None:
        model = model.to(device=trainer.args.device)
    else:
        model = model.to(device=trainer.args.device, dtype=target_dtype)

    model.eval()

    if model is not trainer.model:
        trainer.model_wrapped = model

    return model

def _infer_pred_spans(
    trainer: Trainer,
    dataset: CharBioDataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> Tuple[Dict[str, List[str]], Dict[str, List[Tuple[int, int, str]]]]:
    if len(dataset) == 0:
        return {}, {}

    processor = trainer.processing_class if trainer.processing_class is not None else trainer.tokenizer
    collator = DataCollatorForTokenClassification(processor)

    local_pids, local_indices = _local_pid_partition(dataset)
    local_dataset = torch.utils.data.Subset(dataset, local_indices)

    char_len = dataset.char_lengths
    win_total: Dict[str, int] = Counter()
    for idx in local_indices:
        win_total[dataset.windows[idx].pid] += 1
    win_seen: Dict[str, int] = Counter()

    active_pred_scores: Dict[str, np.ndarray] = {}
    active_counts: Dict[str, np.ndarray] = {}
    local_pred_scores: Dict[str, np.ndarray] = {}
    local_counts: Dict[str, np.ndarray] = {}

    def finalize_pid(pid: str) -> None:
        pred_scores = active_pred_scores.pop(pid, None)
        counts = active_counts.pop(pid, None)
        if pred_scores is None or counts is None:
            return
        local_pred_scores[pid] = pred_scores
        local_counts[pid] = counts

    def _collate(features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pids = [f["pid"] for f in features]
        char_maps = [f["char_map"] for f in features]
        core = []
        for f in features:
            d = dict(f)
            for key in (
                "pid", "char_map",
            ):
                d.pop(key, None)
            core.append(d)

        batch = collator(core)
        max_len = int(batch["input_ids"].shape[1])
        char_map_pad = torch.full((len(char_maps), max_len), -1, dtype=torch.long)
        for i, cm in enumerate(char_maps):
            cm_len = min(int(cm.shape[0]), max_len)
            if cm_len > 0:
                char_map_pad[i, :cm_len] = cm[:cm_len]
        batch["pids"] = pids
        batch["char_maps"] = char_map_pad
        return batch

    loader = DataLoader(
        local_dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        collate_fn=_collate,
    )
    model = _prepare_model_for_eval(trainer, loader)

    crf_model = model.module if hasattr(model, "module") else model
    num_labels = len(LABELS)

    with torch.no_grad():
        for batch in loader:
            pids = batch.pop("pids")
            char_maps = batch.pop("char_maps").cpu().numpy()

            model_inputs = {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
            model_inputs = trainer._prepare_inputs(model_inputs)
            outputs = model(**model_inputs)
            logits_t = outputs.logits
            decoded_t = crf_model.decode(
                logits_t, attention_mask=model_inputs.get("attention_mask"),
            )
            decoded = decoded_t.detach().cpu().numpy()

            for i, pid in enumerate(pids):
                if pid not in char_len:
                    continue
                if pid not in active_pred_scores:
                    plen = int(char_len[pid])
                    active_pred_scores[pid] = np.zeros((plen, num_labels), dtype=np.float64)
                    active_counts[pid] = np.zeros((plen,), dtype=np.float64)

                pred_scores = active_pred_scores[pid]
                counts = active_counts[pid]
                cmap = char_maps[i]
                valid = np.where((cmap >= 0) & (cmap < pred_scores.shape[0]))[0]
                if valid.size > 0:
                    ch = cmap[valid]

                    labels = decoded[i, valid].astype(np.int64, copy=False)
                    ok = (labels >= 0) & (labels < num_labels)
                    if np.any(ok):
                        pred_scores[ch[ok], labels[ok]] += 1.0

                    counts[ch] += 1.0

                win_seen[pid] += 1
                if win_seen[pid] >= win_total.get(pid, 0):
                    finalize_pid(pid)

    for pid in list(active_pred_scores.keys()):
        finalize_pid(pid)

    pred_char_bio: Dict[str, List[str]] = {}
    pred_spans: Dict[str, List[Tuple[int, int, str]]] = {}
    o_label_id = LABEL2ID["O"]

    for pid in local_pids:
        plen = char_len[pid]
        pred_scores = local_pred_scores.get(pid)
        counts = local_counts.get(pid)

        if pred_scores is None or counts is None:
            bio_tags = ["O"] * plen
        else:
            bio_ids = np.full((plen,), o_label_id, dtype=np.int32)
            seen_mask = counts > 0
            if np.any(seen_mask):
                bio_ids[seen_mask] = np.argmax(pred_scores[seen_mask], axis=-1).astype(np.int32, copy=False)
            bio_tags = repair_bio([ID2LABEL[int(idx)] for idx in bio_ids.tolist()])

        pred_char_bio[pid] = bio_tags
        pred_spans[pid] = bio_to_spans(bio_tags)

    return pred_char_bio, pred_spans

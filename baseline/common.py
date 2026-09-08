"""Shared BIO data, CRF model, configuration and runtime setup."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from transformers import (
    AutoConfig, AutoModelForTokenClassification, AutoTokenizer,
    DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput

LABELS = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-BOOK", "I-BOOK"]
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
TYPES = ["PER", "LOC", "BOOK"]
BACKBONES = {
    "sillokbert": ("ddokbaro/SillokBert", "35d95c0555c4157420c46503a378e20b787242e5"),
    "sikuroberta": ("SIKU-BERT/sikuroberta", "bb25260d5c321924fe4fb353c09191c0aaf5c5c6"),
    "classical_chinese": ("KoichiYasuoka/roberta-classical-chinese-base-char",
                          "51e91a5270ce5e68eb31b1c828598c09c3a5e4c6"),
}
def read_rows(path, limit=0, rank=0, world_size=1):
    """Validate public BIO records; keep complete paragraphs on one rank."""
    ids = set()
    with open(path, encoding="utf-8") as handle:
        count = 0
        for line in handle:
            if not line.strip():
                continue
            if limit and count >= limit:
                break
            index = count
            count += 1
            if index % world_size != rank:
                continue
            row = json.loads(line)
            raw = row.get("bio", row.get("char_tags"))
            if not isinstance(raw, list) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}, record {index}: text/bio required")
            if len(row["text"]) != len(raw):
                raise ValueError(f"{path}, record {index}: text/bio length mismatch")
            for offset, item in enumerate(raw):
                tag = item.get("tag") if isinstance(item, dict) else item
                if isinstance(item, dict) and item.get("char") != row["text"][offset]:
                    raise ValueError(f"{path}, record {index}: character mismatch")
                if tag != "O" and (not isinstance(tag, str) or "-" not in tag
                                   or normalize_tag(tag) == "O"):
                    raise ValueError(f"{path}, record {index}: invalid BIO tag {tag!r}")
            pid, text, bio = normalize_row_for_eval(row, index)
            if pid in ids:
                raise ValueError(f"{path}: duplicate id {pid!r}")
            ids.add(pid)
            # Store only the fields used by training/evaluation.
            yield {"id": pid, "text": text, "bio": bio}

@dataclass
class WindowItem:
    pid: str
    input_ids: list
    attention_mask: list
    labels: list
    char_map: list

class CharBioDataset(Dataset):
    """Historical token windows: 510 content tokens, overlap 256."""
    def __init__(self, rows, tokenizer, max_len=510, stride=256):
        if not 0 <= stride < max_len <= 510:
            raise ValueError("Require 0 <= stride < max_len <= 510")
        self.rows = rows
        self.windows = []
        self.char_lengths = {}
        for index, row in enumerate(rows):
            pid, text, bio = normalize_row_for_eval(row, index)
            self.char_lengths[pid] = max(len(text), len(bio))
            enc = tokenizer(list(text), is_split_into_words=True,
                            add_special_tokens=False, return_attention_mask=False)
            try:
                word_ids = enc.word_ids()
            except TypeError:
                word_ids = enc.word_ids(batch_index=0)
            token_labels, token_charidx, seen = [], [], set()
            for wi in word_ids:
                if wi is None or wi in seen:
                    token_labels.append(-100)
                    token_charidx.append(-1)
                else:
                    seen.add(wi)
                    token_labels.append(LABEL2ID[bio[wi]])
                    token_charidx.append(wi)
            ids = enc["input_ids"]
            for start in range(0, len(ids), max_len - stride):
                end = min(len(ids), start + max_len)
                input_ids = [tokenizer.cls_token_id] + ids[start:end] + [tokenizer.sep_token_id]
                self.windows.append(WindowItem(
                    pid, input_ids, [1] * len(input_ids),
                    [-100] + token_labels[start:end] + [-100],
                    [-1] + token_charidx[start:end] + [-1],
                ))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        w = self.windows[index]
        return {"input_ids": torch.tensor(w.input_ids),
                "attention_mask": torch.tensor(w.attention_mask),
                "labels": torch.tensor(w.labels),
                "pid": w.pid, "char_map": torch.tensor(w.char_map)}

class CollatorKeepMeta(DataCollatorForTokenClassification):
    def __call__(self, features):
        core = [{k: v for k, v in f.items() if k not in ("pid", "char_map")}
                for f in features]
        return super().__call__(core)

class PaperTrainer(Trainer):
    def get_train_dataloader(self):
        # Preserve the original sampler/Accelerate setup.
        loader = DataLoader(
            self.train_dataset, batch_size=self._train_batch_size,
            collate_fn=self.data_collator,
            sampler=self._get_train_sampler(),
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            drop_last=self.args.dataloader_drop_last,
        )
        return self.accelerator.prepare(loader)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        model.eval()
        inputs = self._prepare_inputs(inputs)
        labels = inputs.get("labels")
        with torch.no_grad():
            outputs = model(**inputs)
        loss = outputs.loss.detach() if outputs.loss is not None else None
        if prediction_loss_only:
            return loss, None, None
        crf_model = self.accelerator.unwrap_model(model)
        decoded = crf_model.decode(outputs.logits,
                                   attention_mask=inputs.get("attention_mask"), labels=labels)
        return loss, decoded.detach(), labels.detach() if labels is not None else None


def normalize_type(t: str) -> str:
    aliases = {
        "PERSON": "PER",
        "PER": "PER",
        "LOCATION": "LOC",
        "LOC": "LOC",
        "BOOK": "BOOK",
    }
    return aliases.get(str(t), str(t))

def normalize_tag(tag: Optional[str]) -> str:
    if not tag or tag == "O" or "-" not in tag:
        return "O"
    p, t = str(tag).split("-", 1)
    t = normalize_type(t)
    if p not in {"B", "I"} or t not in TYPES:
        return "O"
    return f"{p}-{t}"

def normalize_row_for_eval(r: Dict, i: int) -> Tuple[str, str, List[str]]:
    meta = r.get("meta") or {}
    pid = r.get("id")
    if not pid:
        source_id = meta.get("source_id", "unknown")
        para_idx = meta.get("paragraph_index", -1)
        chunk_idx = meta.get("chunk_index", -1)
        pid = f"{source_id}:{para_idx}:{chunk_idx}:{i}"

    text = r.get("text", "")
    bio = r.get("bio")
    if bio is None:
        bio = r.get("char_tags")
    bio = bio or []
    if bio and isinstance(bio[0], dict):
        bio_tags = [normalize_tag(b.get("tag", "O")) for b in bio]
        if len(text) != len(bio_tags):
            bio_chars = [b.get("char", "") for b in bio]
            text = "".join(bio_chars)
        bio = bio_tags
    else:
        bio = [normalize_tag(tag) for tag in bio]

    return pid, text, bio

def bio_to_spans(bio: List[str]) -> List[Tuple[int,int,str]]:
    spans = []; cur_t=None; cur_s=None
    for i, tag in enumerate(bio):
        if tag == "O" or tag is None or "-" not in tag:
            if cur_t is not None:
                spans.append((cur_s, i, cur_t)); cur_t=None; cur_s=None
            continue
        p, t = tag.split("-",1)
        t = normalize_type(t)
        if p=="B":
            if cur_t is not None: spans.append((cur_s, i, cur_t))
            cur_t=t; cur_s=i
        elif p=="I":
            if cur_t==t:
                pass
            else:
                if cur_t is not None: spans.append((cur_s, i, cur_t))
                cur_t=t; cur_s=i
        else:
            if cur_t is not None:
                spans.append((cur_s, i, cur_t)); cur_t=None; cur_s=None
    if cur_t is not None: spans.append((cur_s, len(bio), cur_t))
    return spans

def repair_bio(tags: List[str]) -> List[str]:
    fixed=[]; prev='O'; prev_type=None
    for t in tags:
        if not t or t=='O' or '-' not in t:
            fixed.append('O'); prev='O'; prev_type=None; continue
        p, typ = t.split('-',1)
        if p=='B':
            fixed.append(t); prev='B'; prev_type=typ
        elif p=='I':
            if prev in ('B','I') and prev_type==typ:
                fixed.append(t); prev='I'
            else:
                fixed.append('B-'+typ); prev='B'; prev_type=typ
        else:
            fixed.append('O'); prev='O'; prev_type=None
    return fixed

def load_token_classification_backbone(model_name_or_path, num_labels, id2label, label2id, revision=None):
    config = AutoConfig.from_pretrained(model_name_or_path, revision=revision)
    checkpoint_num_labels = getattr(config, "num_labels", None)
    ignore_mismatched_sizes = False

    if checkpoint_num_labels not in (None, num_labels):
        ignore_mismatched_sizes = True
        print(
            "[model] checkpoint num_labels="
            f"{checkpoint_num_labels} != current num_labels={num_labels}; "
            "reinitializing token-classification head"
        )

    return AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=ignore_mismatched_sizes, revision=revision,
    )

class TokenClassificationWithCRF(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.crf = CRF(self.config.num_labels, batch_first=True)

    @classmethod
    def from_pretrained(cls, model_name_or_path, num_labels, id2label, label2id, revision=None):
        base = load_token_classification_backbone(
            model_name_or_path,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id, revision=revision,
        )
        return cls(base)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        kwargs.pop("num_items_in_batch", None)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None, **kwargs)
        emissions = outputs.logits
        emissions_f = emissions.float()  # 안정성: CRF는 float32에서 동작시키기
        loss = None
        if labels is not None:
            if attention_mask is not None:
                mask = attention_mask.bool()
            else:
                mask = labels != -100
            if mask.size(1) > 0:
                mask[:, 0] = True
            tags = labels.clone()
            tags = tags.masked_fill(tags == -100, 0)
            loss = -self.crf(emissions_f, tags, mask=mask, reduction="mean")
        return TokenClassifierOutput(
            loss=loss,
            logits=emissions,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, emissions, attention_mask=None, labels=None):
        emissions_f = emissions.float()  # 안정성: CRF decode는 float32
        if labels is not None:
            if attention_mask is not None:
                mask = attention_mask.bool()
            else:
                mask = labels != -100
            if mask.size(1) > 0:
                mask[:, 0] = True
        elif attention_mask is not None:
            mask = attention_mask.bool()
            if mask.size(1) > 0:
                mask[:, 0] = True
        else:
            mask = torch.ones(emissions_f.shape[:2], dtype=torch.bool, device=emissions_f.device)
        decoded = self.crf.decode(emissions_f, mask=mask)
        pred_ids = torch.full(emissions_f.shape[:2], -100, dtype=torch.long, device=emissions_f.device)
        for i, seq in enumerate(decoded):
            idxs = mask[i].nonzero(as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            limit = min(len(seq), idxs.numel())
            if limit == 0:
                continue
            pred_ids[i, idxs[:limit]] = torch.tensor(seq[:limit], device=emissions.device)
        return pred_ids

def load_fast_tokenizer(tokenizer_src: str, revision=None):
    tok = AutoTokenizer.from_pretrained(tokenizer_src, use_fast=True, revision=revision)
    if not getattr(tok, "is_fast", False):
        raise ValueError(
            "This NER pipeline requires a fast tokenizer because CharBioDataset uses word_ids(). "
            "Install sentencepiece/tiktoken or use a tokenizer path that contains tokenizer.json."
        )
    return tok

def reinit_new_embeddings(model, old_vocab_size: int, new_tokens: int) -> None:
    if new_tokens <= 0:
        return
    input_emb = model.get_input_embeddings()
    if input_emb is None:
        return
    std = getattr(model.config, "initializer_range", 0.02)
    with torch.no_grad():
        torch.nn.init.normal_(input_emb.weight[old_vocab_size:], mean=0.0, std=std)

    output_emb = model.get_output_embeddings()
    if output_emb is not None and output_emb.weight.shape[0] == input_emb.weight.shape[0]:
        with torch.no_grad():
            torch.nn.init.normal_(output_emb.weight[old_vocab_size:], mean=0.0, std=std)

def _dist_is_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def _dist_rank() -> int:
    if _dist_is_initialized():
        return int(torch.distributed.get_rank())
    return _env_int("RANK", 0)

def _dist_world_size() -> int:
    if _dist_is_initialized():
        return int(torch.distributed.get_world_size())
    return _env_int("WORLD_SIZE", 1)

def _env_int(name: str, default: int = -1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default

def seen_entity_texts(rows: List[Dict]) -> Dict[str, set]:
    seen = {typ: set() for typ in TYPES}
    for i, row in enumerate(rows):
        pid, text, bio = normalize_row_for_eval(row, i)
        del pid
        for start, end, typ in bio_to_spans(bio):
            if typ in seen:
                seen[typ].add(text[start:end])
    return seen

def save_json(path, value):
    def convert(obj):
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.detach().cpu().tolist()
        raise TypeError(type(obj).__name__)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=convert)
        handle.write("\n")

def parse_args(evaluation_only=False):
    parser = argparse.ArgumentParser(description=(
        "Evaluate a saved BIO + CRF checkpoint without training." if evaluation_only
        else "Train BIO + CRF, select the best dev checkpoint and evaluate test."))
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--dev_jsonl", required=True)
    parser.add_argument("--test_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", choices=tuple(BACKBONES), default="sillokbert")
    parser.add_argument("--model_name_or_path", help="Override with a local model or Hub ID")
    parser.add_argument("--revision", help="Override the recorded backbone revision")
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=512)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_len", type=int, default=510)
    parser.add_argument("--stride", type=int, default=256, help="Window overlap in tokens")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--seen_reference", choices=("train+dev", "train"), default="train+dev")
    parser.add_argument("--test_chunk_rows", type=int, default=10000)
    parser.add_argument("--test_only", action="store_true")
    parser.add_argument("--checkpoint", required=evaluation_only,
                        help="CRF checkpoint directory; required for evaluation")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1, help="Smoke tests only")
    parser.add_argument("--limit_train_rows", type=int, default=0)
    parser.add_argument("--limit_dev_rows", type=int, default=0)
    parser.add_argument("--limit_test_rows", type=int, default=0)
    args = parser.parse_args()
    if evaluation_only:
        args.test_only = True
    if args.test_only and not args.checkpoint:
        parser.error("--test_only requires --checkpoint")
    if args.checkpoint and not args.test_only:
        parser.error("--checkpoint is an evaluation-only option")
    if args.use_cpu and args.precision != "fp32":
        parser.error("CPU smoke tests require --precision fp32")
    if args.test_chunk_rows < 1 or min(args.limit_train_rows, args.limit_dev_rows, args.limit_test_rows) < 0:
        parser.error("Invalid chunk size or row limit")
    return args

def build_runtime(args):
    """Initialize identical model/data/Trainer settings for train or evaluation."""
    output_dir = Path(args.output_dir)
    # Fail before overwriting any existing experiment.
    if _env_int("RANK", 0) == 0:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"Use an empty output_dir: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    source, revision = BACKBONES[args.model]
    if args.model_name_or_path:
        source, revision = args.model_name_or_path, None
    revision = args.revision or revision
    tok = load_fast_tokenizer(source, revision=revision)
    if "▣" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["▣"]})
    model = TokenClassificationWithCRF.from_pretrained(
        source, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        revision=revision)
    old_vocab = model.model.get_input_embeddings().weight.shape[0]
    if len(tok) > old_vocab:
        model.model.resize_token_embeddings(len(tok))
        reinit_new_embeddings(model.model, old_vocab, len(tok) - old_vocab)
    if args.checkpoint:
        from safetensors.torch import load_file
        checkpoint = Path(args.checkpoint)
        state = checkpoint / "model.safetensors"
        weights = load_file(str(state)) if state.exists() else torch.load(
            checkpoint / "pytorch_model.bin", map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)

    train_rows = list(read_rows(args.train_jsonl, args.limit_train_rows))
    dev_rows = list(read_rows(args.dev_jsonl, args.limit_dev_rows))
    if not train_rows or not dev_rows:
        raise ValueError("Nonempty train and dev files are required")
    seen = seen_entity_texts(train_rows)
    if args.seen_reference == "train+dev":
        dev_seen = seen_entity_texts(dev_rows)
        for typ in TYPES:
            seen[typ].update(dev_seen[typ])
    train_ds = None if args.test_only else CharBioDataset(train_rows, tok, args.max_len, args.stride)
    dev_ds = None if args.test_only else CharBioDataset(dev_rows, tok, args.max_len, args.stride)
    kwargs = dict(
        output_dir=str(output_dir), per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr, weight_decay=0.01, warmup_ratio=0.1,
        num_train_epochs=args.epochs, max_steps=args.max_steps,
        optim="adamw_torch", lr_scheduler_type="linear",
        adam_beta1=0.9, adam_beta2=0.999, adam_epsilon=1e-8, max_grad_norm=1.0,
        eval_strategy="no" if args.test_only else "epoch",
        save_strategy="no" if args.test_only else "epoch", save_total_limit=1,
        logging_strategy="steps", logging_steps=100,
        fp16=args.precision == "fp16", bf16=args.precision == "bf16",
        seed=args.seed, report_to=[], load_best_model_at_end=not args.test_only,
        metric_for_best_model="eval_micro_f1", greater_is_better=True,
        remove_unused_columns=False, eval_accumulation_steps=2,
        dataloader_num_workers=args.num_workers, dataloader_pin_memory=not args.use_cpu,
        ddp_timeout=72000, use_cpu=args.use_cpu,
    )
    targs = TrainingArguments(**kwargs)

    trainer = PaperTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=dev_ds,
        data_collator=CollatorKeepMeta(tok), processing_class=tok,
        compute_metrics=None,
    )
    # Some Transformers versions log scalar tensors in TrainerState.
    # Convert only values being serialized, preserving numeric JSON fields.
    from transformers.trainer_callback import TrainerState
    import dataclasses
    def state_to_json(state, path):
        save_json(path, dataclasses.asdict(state))
    TrainerState.save_to_json = state_to_json

    if trainer.is_world_process_zero():
        import importlib.metadata
        config = dict(vars(args), backbone_source=source, backbone_revision=revision,
                      world_size=targs.world_size,
                      effective_batch_size=args.train_batch_size * targs.world_size *
                      args.gradient_accumulation_steps,
                      versions={name: importlib.metadata.version(name) for name in
                                ("torch", "transformers", "tokenizers", "accelerate", "pytorch-crf")})
        save_json(output_dir / "run_config.json", config)
        model.config.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)
        print("[config]", json.dumps(config), flush=True)
    return SimpleNamespace(
        trainer=trainer, tokenizer=tok, train_rows=train_rows, dev_rows=dev_rows,
        train_ds=train_ds, dev_ds=dev_ds, seen=seen,
    )


def release_datasets(runtime):
    """Release training/dev windows before chunked full-corpus evaluation."""
    import gc
    runtime.trainer.compute_metrics = None
    runtime.trainer.train_dataset = runtime.trainer.eval_dataset = None
    runtime.train_rows = runtime.dev_rows = runtime.train_ds = runtime.dev_ds = None
    gc.collect()
    if torch.cuda.is_available() and not runtime.trainer.args.use_cpu:
        torch.cuda.empty_cache()

"""CPU regression checks; no network, production data, or GPU required."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, mock_open, patch

import numpy as np
import torch
from transformers import BertConfig, BertForTokenClassification, BertTokenizerFast
from baseline import common as paper
from baseline import evaluate, inference, train


def fixture(root):
    model_dir = root / "tiny-model"
    model_dir.mkdir()
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "李", "氏", "京", "城", "書", "。", "▣"]
    (model_dir / "vocab.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
    tokenizer = BertTokenizerFast(vocab_file=str(model_dir / "vocab.txt"), do_lower_case=False)
    tokenizer.save_pretrained(model_dir)
    config = BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, num_labels=7,
                        id2label=paper.ID2LABEL, label2id=paper.LABEL2ID,
                        max_position_embeddings=512)
    BertForTokenClassification(config).save_pretrained(model_dir)
    rows = [{"id": str(i), "text": "李氏京城書。", "bio": [
        {"char": c, "tag": t} for c, t in zip("李氏京城書。",
        ["B-PER", "I-PER", "B-LOC", "I-LOC", "B-BOOK", "O"])]} for i in range(8)]
    for split in ("train", "dev", "test"):
        (root / (split + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    return model_dir, tokenizer, rows


class RegressionTests(unittest.TestCase):
    def test_cli_entrypoints(self):
        repo = Path(__file__).resolve().parent.parent
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HF_HUB_OFFLINE="1",
                   PYTHONDONTWRITEBYTECODE="1")
        for entry in ([str(repo / "baseline/train.py")],
                      [str(repo / "baseline/evaluate.py")],
                      ["-m", "baseline.train"], ["-m", "baseline.evaluate"]):
            with self.subTest(entry=entry):
                result = subprocess.run([sys.executable, *entry, "--help"], cwd=repo,
                                        env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("--test_jsonl", result.stdout)

    def test_input_validation_and_required_loaders(self):
        for tag in ("B-INVALID", "X-PER", "PER", None, 1):
            row = {"id": "a", "text": "李", "bio": [{"char": "李", "tag": tag}]}
            with self.subTest(tag=tag), patch("builtins.open", mock_open(read_data=json.dumps(row))):
                with self.assertRaisesRegex(ValueError, "invalid BIO tag"):
                    list(paper.read_rows("fixture.jsonl"))
        row = {"id": "a", "text": "李", "bio": [{"char": "李", "tag": "B-PERSON"}]}
        with patch("builtins.open", mock_open(read_data=json.dumps(row))):
            self.assertEqual(list(paper.read_rows("fixture.jsonl"))[0]["bio"], ["B-PER"])
        with patch.object(paper.AutoTokenizer, "from_pretrained", return_value=Mock(is_fast=True)) as load:
            paper.load_fast_tokenizer("fixture", revision="revision")
            load.assert_called_once_with("fixture", use_fast=True, revision="revision")
        with patch.object(paper.AutoTokenizer, "from_pretrained", return_value=Mock(is_fast=False)):
            with self.assertRaisesRegex(ValueError, "requires a fast tokenizer"):
                paper.load_fast_tokenizer("fixture")
        with patch.object(paper.AutoTokenizer, "from_pretrained", side_effect=ValueError("broken")) as load:
            with self.assertRaisesRegex(ValueError, "broken"):
                paper.load_fast_tokenizer("fixture")
            self.assertEqual(load.call_count, 1)
        with patch.object(paper.AutoConfig, "from_pretrained", side_effect=OSError("bad config")):
            with self.assertRaisesRegex(OSError, "bad config"):
                paper.load_token_classification_backbone("fixture", 7, paper.ID2LABEL, paper.LABEL2ID)

    def test_metric_partition_and_streaming(self):
        rows = [{"id": "x", "text": "李京書", "bio": ["B-PER", "B-LOC", "B-BOOK"]},
                {"id": "y", "text": "李京書", "bio": ["B-PER", "B-LOC", "B-BOOK"]}]
        seen = {"PER": {"李"}, "LOC": set(), "BOOK": {"書"}}
        pred = {"x": [(0, 1, "PER"), (1, 2, "PER"), (2, 3, "BOOK")], "y": []}
        gold = {r["id"]: paper.bio_to_spans(r["bio"]) for r in rows}
        texts = {r["id"]: r["text"] for r in rows}
        expected = evaluate.core_metrics(gold, pred, texts, seen)
        accumulator = evaluate.MetricAccumulator(seen)
        for row in rows:
            accumulator.update([row], pred)
        self.assertEqual(accumulator.results(), expected)
        self.assertAlmostEqual(expected["seen_f1"], 2 / 3)
        self.assertEqual(expected["unseen_f1"], 0)
        self.assertFalse(any(key.endswith("_support") for key in expected))
        self.assertEqual(expected["error_type_error"], 1)
        self.assertEqual(expected["error_missing"], 3)

    def test_error_diagnostics_and_empty_inputs(self):
        gold = {"a": [(0, 2, "PER"), (3, 5, "LOC"), (6, 7, "BOOK")]}
        pred = {"a": [(0, 1, "PER"), (3, 5, "BOOK"), (8, 9, "LOC")]}
        metrics = evaluate.core_metrics(gold, pred, {"a": "李氏京城書文字天地"}, {})
        for name in ("boundary_error", "type_error", "missing", "spurious"):
            self.assertEqual(metrics["error_" + name], 1)
        self.assertEqual(metrics["micro_f1"], 0)
        self.assertTrue(all(v == 0 for v in evaluate.core_metrics({}, {}, {}, {}).values()))
        # Equal overlap prefers matching type; otherwise preserve prediction order.
        counts = evaluate.count_span_errors([(0, 2, "PER")],
                                         [(0, 1, "LOC"), (1, 2, "PER")])
        self.assertEqual(counts["boundary_error"], 1)
        self.assertEqual(counts["spurious"], 1)

    def test_windows_partition_and_crf(self):
        with tempfile.TemporaryDirectory(prefix="ner-windows-") as directory:
            root = Path(directory)
            _, tokenizer, rows = fixture(root)
            dataset = paper.CharBioDataset(rows, tokenizer, max_len=4, stride=2)
            self.assertEqual(dataset.char_lengths, {r["id"]: len(r["text"]) for r in rows})
            self.assertEqual(len(dataset), 3 * len(rows))
            for w in dataset.windows:
                self.assertEqual(len(w.input_ids), len(w.labels))
                self.assertEqual(len(w.input_ids), len(w.char_map))
                self.assertEqual(w.char_map[0], -1)
                self.assertEqual(w.char_map[-1], -1)
            with patch.object(inference, "_dist_world_size", return_value=2), \
                 patch.object(inference, "_dist_rank", return_value=1):
                pids, indices = inference._local_pid_partition(dataset)
                self.assertEqual(pids, [r["id"] for r in rows[1::2]])
                self.assertEqual(indices, [i for i, w in enumerate(dataset.windows) if w.pid in pids])
                dataset.distributed_prepartitioned = True
                self.assertEqual(inference._local_pid_partition(dataset),
                                 ([r["id"] for r in rows], list(range(len(dataset)))))
            labels = np.full((len(dataset), 6), -100, dtype=np.int64)
            for i, w in enumerate(dataset.windows):
                labels[i, :len(w.labels)] = w.labels
            bios, spans = inference.merge_and_decode(
                dataset, labels, [w.pid for w in dataset.windows],
                [torch.tensor(w.char_map) for w in dataset.windows])
            for row in rows:
                tags = [item["tag"] for item in row["bio"]]
                self.assertEqual(bios[row["id"]], tags)
                self.assertEqual(spans[row["id"]], paper.bio_to_spans(tags))
            config = BertConfig(vocab_size=len(tokenizer), hidden_size=16,
                                num_hidden_layers=1, num_attention_heads=2,
                                intermediate_size=32, num_labels=7)
            model = paper.TokenClassificationWithCRF(BertForTokenClassification(config))
            model.eval()
            batch = paper.CollatorKeepMeta(tokenizer)([dataset[0], dataset[2]])
            outputs = model(**batch)
            self.assertTrue(torch.isfinite(outputs.loss).item())
            outputs.loss.backward()
            decoded = model.decode(outputs.logits.detach(), batch["attention_mask"])
            self.assertEqual(decoded.shape, batch["input_ids"].shape)

    def test_cpu_training_and_checkpoint_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="ner-paper-smoke-") as directory:
            root = Path(directory)
            model, _, rows = fixture(root)
            common = [str(Path(train.__file__).resolve()), "--model_name_or_path", str(model),
                      "--train_jsonl", str(root / "train.jsonl"),
                      "--dev_jsonl", str(root / "dev.jsonl"),
                      "--test_jsonl", str(root / "test.jsonl"),
                      "--use_cpu", "--precision", "fp32", "--num_workers", "0",
                      "--train_batch_size", "2", "--eval_batch_size", "2",
                      "--test_chunk_rows", "3", "--max_len", "4", "--stride", "2"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1")
            for name, extra in (("train-run", ["--max_steps", "2"]),
                                ("reload-run", ["--checkpoint",
                                                str(root / "train-run" / "best_model")])):
                entry = [str(Path(evaluate.__file__).resolve()), *common[1:]] if name == "reload-run" else common
                result = subprocess.run([sys.executable, *entry, "--output_dir", str(root / name), *extra],
                                        env=env, text=True, stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT, timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout[-16000:])
                predictions = [json.loads(line) for line in
                               (root / name / "test_predictions.jsonl").read_text().splitlines()]
                self.assertEqual([r["id"] for r in predictions], [r["id"] for r in rows])
            self.assertEqual((root / "train-run" / "test_metrics.json").read_text(),
                             (root / "reload-run" / "test_metrics.json").read_text())
            dev = json.loads((root / "train-run" / "dev_metrics.json").read_text())
            test = json.loads((root / "train-run" / "test_metrics.json").read_text())
            self.assertEqual({k: dev[k] for k in evaluate.CORE_METRIC_KEYS}, test)
            # An odd paragraph count exercises unequal rank lengths and the
            # order-preserving merge. Include training to exercise DDP saving.
            result = subprocess.run([
                sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2",
                *common, "--output_dir", str(root / "ddp-run"), "--max_steps", "2",
                "--limit_test_rows", "7"], env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout[-16000:])
            predictions = [json.loads(line) for line in
                           (root / "ddp-run" / "test_predictions.jsonl").read_text().splitlines()]
            self.assertEqual([r["id"] for r in predictions], [r["id"] for r in rows[:7]])
            gold = {r["id"]: paper.bio_to_spans([v["tag"] for v in r["bio"]]) for r in rows[:7]}
            expected = evaluate.core_metrics(gold, {r["id"]: r["pred_spans"] for r in predictions},
                                          {r["id"]: r["text"] for r in rows[:7]},
                                          paper.seen_entity_texts(rows))
            self.assertEqual(json.loads((root / "ddp-run" / "test_metrics.json").read_text()), expected)


if __name__ == "__main__":
    unittest.main()

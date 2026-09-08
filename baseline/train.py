#!/usr/bin/env python3
"""Train one backbone + CRF, select by dev F1, then evaluate test."""
from __future__ import annotations

# Support both "python baseline/file.py" and "python -m baseline.file".
if not __package__:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "baseline"

from pathlib import Path

from .common import parse_args, build_runtime, release_datasets, save_json
from .evaluate import make_dev_metrics, evaluate_test


def main():
    args = parse_args()
    runtime = build_runtime(args)
    trainer, tok = runtime.trainer, runtime.tokenizer
    output_dir = Path(args.output_dir)
    if not args.test_only:
        trainer.compute_metrics = make_dev_metrics(runtime.dev_ds, runtime.dev_rows, runtime.seen)
        trainer.train()
        dev_results = trainer.evaluate()
        if trainer.is_world_process_zero():
            save_json(output_dir / "dev_metrics.json", {
                k.removeprefix("eval_"): v for k, v in dev_results.items()})
        trainer.save_model(str(output_dir / "best_model"))
        if trainer.is_world_process_zero():
            trainer.model.config.save_pretrained(output_dir / "best_model")
            save_json(output_dir / "best_checkpoint.json", {
                "checkpoint": trainer.state.best_model_checkpoint,
                "metric": trainer.state.best_metric,
                "global_step": trainer.state.global_step})
    release_datasets(runtime)
    evaluate_test(trainer, tok, args, runtime.seen)


if __name__ == "__main__":
    main()

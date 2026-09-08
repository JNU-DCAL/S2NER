"""Validate release shell commands without starting GPU jobs."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


class LauncherTests(unittest.TestCase):
    def test_dry_runs_and_invalid_arguments(self):
        baseline = Path(__file__).resolve().parent.parent / "baseline"
        with tempfile.TemporaryDirectory(prefix="ner-launcher-") as directory:
            root = Path(directory)
            data = root / "data with spaces"
            data.mkdir()
            for name in ("SJW.jsonl", "Sillok_train_final.jsonl", "Sillok_dev_final.jsonl"):
                (data / name).touch()
            env = dict(os.environ, DATA_DIR=str(data), GPUS="0,1,2,3", DRY_RUN="1",
                       OUTPUT_ROOT=str(root / "outputs"), PYTHON_BIN="/unused/python with spaces")
            for script, expected in (
                ("run_sillokbert.sh", ["sillokbert"]),
                ("run_classical_chinese.sh", ["classical_chinese"]),
                ("run_siku.sh", ["sikuroberta"]),
                ("run_experiments.sh", ["sikuroberta", "sillokbert", "classical_chinese"]),
            ):
                with self.subTest(script=script):
                    subprocess.run(["bash", "-n", str(baseline / script)], check=True)
                    run = subprocess.run(["bash", str(baseline / script)], env=env,
                                         capture_output=True, text=True, timeout=10)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    commands = [shlex.split(line.removeprefix("Running: "))
                                for line in run.stdout.splitlines()]
                    self.assertEqual([c[c.index("--model") + 1] for c in commands], expected)
                    for command in commands:
                        for key, value in (("--lr", "6e-5"), ("--epochs", "4"),
                                           ("--train_batch_size", "128"), ("--precision", "bf16"),
                                           ("--seen_reference", "train+dev"),
                                           ("--test_jsonl", str(data / "SJW.jsonl"))):
                            self.assertEqual(command[command.index(key) + 1], value)
                        self.assertIn("--nproc_per_node=4", command)
                        self.assertEqual(command[0], env["PYTHON_BIN"])
            self.assertFalse((root / "outputs").exists())
            for override, args in (({"GPUS": "0,0,1,2"}, []), ({"GPUS": "0,1"}, []),
                                   ({"DRY_RUN": "wrong"}, []), ({}, ["unknown"]),
                                   ({"DATA_DIR": str(root / "missing")}, [])):
                run = subprocess.run(["bash", str(baseline / "run_experiments.sh"), *args],
                                     env=dict(env, **override), capture_output=True, timeout=10)
                self.assertNotEqual(run.returncode, 0)


if __name__ == "__main__":
    unittest.main()

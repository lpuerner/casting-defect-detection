"""Run every method on both splits, then build the comparison table.

Run:  python run_all.py            (full run, ~1-1.5 h on a 2-core laptop CPU)
      python run_all.py --smoke    (tiny check, ~3 min)
"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
SCRIPTS = ["baseline.py", "train_cnn.py", "transfer.py", "anomaly.py", "ensemble.py"]  # ensemble last: it reads the others' scores
# The clean split is small (910 training images), so it gets more epochs than the official one.
EXTRA = {("train_cnn.py", "clean"): ["--epochs", "30"], ("transfer.py", "clean"): ["--epochs", "8"]}


def main() -> None:
    smoke = ["--smoke"] if "--smoke" in sys.argv else []
    for split in ["clean", "official"]:
        for script in SCRIPTS:
            cmd = [sys.executable, str(SRC / script), "--split", split, *smoke, *EXTRA.get((script, split), [])]
            print("\n>>>", " ".join(cmd[1:]), flush=True)
            subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(SRC / "evaluate.py"), *smoke], check=True)


if __name__ == "__main__":
    main()

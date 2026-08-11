from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils import load_config


def run(cmd):
    print("\n>>>", " ".join(map(str, cmd)))
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-external", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    py = sys.executable
    q = ["--quick"] if args.quick else []

    manifest = Path(cfg["data"]["manifest_csv"])
    if not manifest.exists():
        run([py, "experiments/prepare_data.py", "--config", args.config])

    run([py, "experiments/primary_binary.py", "--config", args.config, *q])
    run([py, "experiments/separability_15class.py", "--config", args.config])
    run([py, "experiments/ablation.py", "--config", args.config, *q])
    run([py, "experiments/optimizer_benchmark.py", "--config", args.config, *q])
    run([py, "experiments/modern_baselines.py", "--config", args.config, *q])
    run([py, "experiments/calibration_stats.py", "--config", args.config])
    run([py, "experiments/attention_errors.py", "--config", args.config])
    run([py, "experiments/complexity_runtime.py", "--config", args.config])
    run([py, "experiments/verify_all.py", "--config", args.config])

    if not args.skip_external and cfg["data"].get("external_root"):
        checkpoint = str(Path(cfg["project"]["output_dir"]) / "best_model.pt")
        run([
            py, "experiments/external_validation.py",
            "--config", args.config,
            "--checkpoint", checkpoint,
            "--external-root", str(cfg["data"]["external_root"]),
        ])


if __name__ == "__main__":
    main()

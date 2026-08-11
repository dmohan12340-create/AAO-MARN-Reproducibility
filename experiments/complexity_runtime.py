from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model import model_from_config
from src.utils import ensure_dir, load_config, resolve_device, save_json


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def benchmark(model, device, image_size, warmup, iterations):
    x = torch.rand(1, 3, image_size, image_size, device=device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return {
        "latency_ms_per_image": 1000.0 * elapsed / iterations,
        "throughput_images_per_second": iterations / elapsed,
        "peak_cuda_memory_bytes": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = ensure_dir(cfg["project"]["output_dir"])
    checkpoint = torch.load(out / "best_model.pt", map_location="cpu")
    model = model_from_config(checkpoint["config"], checkpoint["selected_channels"])
    model.load_state_dict(checkpoint["state_dict"])
    device = resolve_device(cfg["runtime"].get("device", "auto"))
    model = model.to(device)
    total, trainable = count_parameters(model)
    stats = benchmark(
        model,
        device,
        int(cfg["data"]["image_size"]),
        int(cfg["runtime"].get("benchmark_warmup", 10)),
        int(cfg["runtime"].get("benchmark_iterations", 50)),
    )
    stats.update(
        {
            "parameters_total": int(total),
            "parameters_trainable": int(trainable),
            "selected_vgg_channels": len(checkpoint["selected_channels"]),
            "theoretical_terms": {
                "CLAHE": "O(N)",
                "SLIC": "approximately O(N) per local assignment iteration",
                "KMeans": "O(N*K*I)",
                "GOA": "O(P*G*D)",
                "MHA": "O(L^2*d)",
                "RNN": "O(L*d*h + L*h^2)",
                "AAO": "O(P_A*I_A*C_objective)",
            },
        }
    )
    save_json(stats, out / "complexity_runtime.json")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

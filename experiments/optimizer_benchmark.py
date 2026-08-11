from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.primary_binary import train_once
from src.data_pipeline import load_manifest
from src.optimizers import artificial_algae_search, bayesian_search, ga_search, gwo_search, pso_search
from src.utils import ensure_dir, load_config, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = ensure_dir(cfg["project"]["output_dir"])
    manifest = load_manifest(cfg["data"]["manifest_csv"])
    selected_path = out / "goa_selected_channels.json"
    selected = list(range(512))
    if selected_path.exists():
        selected = json.loads(selected_path.read_text())["selected_channels"]

    a = cfg["aao"]
    population = int(a["quick_population"] if args.quick else a["population"])
    iterations = int(a["quick_iterations"] if args.quick else a["iterations"])
    budget = population * iterations
    seed = int(cfg["project"]["seed"])

    cache = {}
    def objective(params):
        key = json.dumps(params, sort_keys=True)
        if key not in cache:
            _, _, _, _, _, val_score = train_once(
                cfg, manifest, selected, seed, quick=True, param_override=params
            )
            cache[key] = float(val_score)
        return cache[key]

    searchers = {
        "AAO": lambda: artificial_algae_search(
            objective, a["search_space"], population, iterations,
            patience=int(a.get("patience", 5)),
            min_delta=float(a.get("min_delta", 0.0005)),
            adaptation_probability=float(a.get("adaptation_probability", 0.3)),
            seed=seed,
        ),
        "PSO": lambda: pso_search(objective, a["search_space"], population, iterations, seed),
        "GA": lambda: ga_search(objective, a["search_space"], population, iterations, seed),
        "GWO": lambda: gwo_search(objective, a["search_space"], population, iterations, seed),
        "BayesianOptimization": lambda: bayesian_search(objective, a["search_space"], budget, seed),
    }

    rows = []
    for name, fn in searchers.items():
        result = fn()
        rows.append(
            {
                "optimizer": name,
                "best_validation_balanced_accuracy": result.best_score,
                "runtime_seconds": result.runtime_seconds,
                "objective_budget_nominal": budget,
                "best_params": json.dumps(result.best_params, sort_keys=True),
            }
        )
        save_json(
            {"best_params": result.best_params, "best_score": result.best_score, "history": result.history},
            out / f"optimizer_{name}.json",
        )

    # Conventional Adam/AdamW is represented by fixed configured hyperparameters.
    _, _, _, _, _, fixed_score = train_once(cfg, manifest, selected, seed, quick=True, param_override={})
    rows.append(
        {
            "optimizer": cfg["training"].get("optimizer", "adamw"),
            "best_validation_balanced_accuracy": fixed_score,
            "runtime_seconds": float("nan"),
            "objective_budget_nominal": 1,
            "best_params": json.dumps(
                {
                    "learning_rate": cfg["training"]["learning_rate"],
                    "weight_decay": cfg["training"]["weight_decay"],
                    "dropout": cfg["model"]["dropout"],
                },
                sort_keys=True,
            ),
        }
    )

    pd.DataFrame(rows).to_csv(out / "optimizer_benchmark.csv", index=False)
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from .utils import save_json


@dataclass
class SearchResult:
    best_params: Dict
    best_score: float
    history: List[Dict]
    runtime_seconds: float


def _repair_mask(
    mask: np.ndarray, min_selected: int, max_selected: int, rng: np.random.Generator
) -> np.ndarray:
    mask = mask.astype(bool).copy()
    n = int(mask.sum())
    if n < min_selected:
        off = np.where(~mask)[0]
        choose = rng.choice(off, size=min(min_selected - n, len(off)), replace=False)
        mask[choose] = True
    if int(mask.sum()) > max_selected:
        on = np.where(mask)[0]
        keep = rng.choice(on, size=max_selected, replace=False)
        mask[:] = False
        mask[keep] = True
    return mask


def goa_feature_selection(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    population: int = 20,
    generations: int = 25,
    alpha: float = 0.95,
    beta: float = 0.05,
    min_selected: int = 32,
    max_selected: int = 256,
    c_max: float = 1.0,
    c_min: float = 0.00004,
    seed: int = 17,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Binary GOA over feature channels.

    Candidate position x in [0,1]^D is thresholded at 0.5.
    Fitness = alpha * validation balanced accuracy
            + beta  * feature reduction ratio.
    """
    if not np.isclose(alpha + beta, 1.0):
        raise ValueError("GOA alpha + beta must equal 1.")
    rng = np.random.default_rng(seed)
    d = x_train.shape[1]

    scaler = StandardScaler()
    xtr = scaler.fit_transform(x_train)
    xva = scaler.transform(x_val)

    pop = rng.random((population, d))
    best_pos = pop[0].copy()
    best_score = -np.inf
    history = []
    cache = {}

    def evaluate(pos: np.ndarray):
        mask = _repair_mask(pos >= 0.5, min_selected, min(max_selected, d), rng)
        key = np.packbits(mask.astype(np.uint8)).tobytes()
        if key in cache:
            return cache[key]
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            random_state=seed,
        )
        clf.fit(xtr[:, mask], y_train)
        pred = clf.predict(xva[:, mask])
        bal = balanced_accuracy_score(y_val, pred)
        reduction = 1.0 - float(mask.sum()) / float(d)
        fit = alpha * bal + beta * reduction
        cache[key] = (fit, bal, reduction, mask.copy())
        return cache[key]

    for g in range(generations):
        c = c_max - g * ((c_max - c_min) / max(generations - 1, 1))

        scores = []
        for i in range(population):
            fit, bal, reduction, mask = evaluate(pop[i])
            scores.append(fit)
            if fit > best_score:
                best_score = fit
                best_pos = pop[i].copy()
                best_mask = mask.copy()

        history.append(
            {
                "generation": g,
                "best_fitness": float(best_score),
                "mean_fitness": float(np.mean(scores)),
                "selected_channels": int(best_mask.sum()),
                "c": float(c),
            }
        )

        # Continuous GOA-inspired social update around current best.
        new_pop = np.empty_like(pop)
        for i in range(population):
            social = np.zeros(d)
            for j in range(population):
                if i == j:
                    continue
                diff = pop[j] - pop[i]
                dist = np.linalg.norm(diff) + 1e-9
                # attraction/repulsion social interaction
                s = 0.5 * np.exp(-dist / 1.5) - np.exp(-dist)
                social += s * diff / dist
            stochastic = rng.normal(0, 0.04, d)
            new_pop[i] = best_pos + c * social / max(population - 1, 1) + stochastic
        pop = np.clip(new_pop, 0.0, 1.0)

    return np.where(best_mask)[0].astype(int), history


def _sample_from_space(rng: np.random.Generator, space: Dict) -> Dict:
    out = {}
    for k, bounds in space.items():
        lo, hi = bounds
        if k in {"attention_dim", "recurrent_hidden", "attention_heads"}:
            out[k] = int(rng.integers(int(lo), int(hi) + 1))
        elif k in {"learning_rate", "weight_decay"}:
            out[k] = float(np.exp(rng.uniform(np.log(float(lo)), np.log(float(hi)))))
        else:
            out[k] = float(rng.uniform(float(lo), float(hi)))
    return repair_hparams(out, space)


def repair_hparams(params: Dict, space: Dict) -> Dict:
    p = dict(params)
    for k, (lo, hi) in space.items():
        p[k] = min(max(p[k], lo), hi)

    p["attention_heads"] = int(round(p.get("attention_heads", 4)))
    p["attention_dim"] = int(round(p.get("attention_dim", 256)))
    p["recurrent_hidden"] = int(round(p.get("recurrent_hidden", 192)))

    heads = max(1, p["attention_heads"])
    dim = max(heads, p["attention_dim"])
    dim = int(round(dim / heads) * heads)
    lo, hi = space["attention_dim"]
    if dim < lo:
        dim = int(math.ceil(lo / heads) * heads)
    if dim > hi:
        dim = int(math.floor(hi / heads) * heads)
    p["attention_dim"] = max(heads, dim)
    return p


def _params_to_vector(params: Dict, keys: Sequence[str]) -> np.ndarray:
    return np.asarray([float(params[k]) for k in keys], dtype=float)


def _vector_to_params(vec: np.ndarray, keys: Sequence[str], space: Dict) -> Dict:
    return repair_hparams({k: float(v) for k, v in zip(keys, vec)}, space)


def artificial_algae_search(
    objective: Callable[[Dict], float],
    search_space: Dict,
    population: int = 12,
    iterations: int = 15,
    patience: int = 5,
    min_delta: float = 5e-4,
    adaptation_probability: float = 0.30,
    seed: int = 17,
) -> SearchResult:
    """
    Practical bounded AAO implementation with:
      - algae colonies as candidate hyperparameters;
      - tournament neighbor selection;
      - helical perturbation;
      - starvation tracking;
      - adaptive reset toward the best colony;
      - explicit early stopping.
    """
    rng = np.random.default_rng(seed)
    keys = list(search_space)
    colonies = [_sample_from_space(rng, search_space) for _ in range(population)]
    starvation = np.zeros(population, dtype=int)
    history: List[Dict] = []
    best_params = None
    best_score = -np.inf
    no_improve = 0
    start = time.perf_counter()

    lo = np.array([search_space[k][0] for k in keys], dtype=float)
    hi = np.array([search_space[k][1] for k in keys], dtype=float)
    scale = np.maximum(hi - lo, 1e-12)

    for iteration in range(iterations):
        scores = np.array([objective(c) for c in colonies], dtype=float)
        order = np.argsort(scores)[::-1]

        if scores[order[0]] > best_score + min_delta:
            best_score = float(scores[order[0]])
            best_params = dict(colonies[order[0]])
            no_improve = 0
        else:
            no_improve += 1

        current_best = _params_to_vector(colonies[order[0]], keys)
        new_colonies = []

        for i, colony in enumerate(colonies):
            current = _params_to_vector(colony, keys)

            # Tournament-selection neighbor.
            contenders = rng.choice(population, size=min(3, population), replace=False)
            neighbor_idx = int(contenders[np.argmax(scores[contenders])])
            neighbor = _params_to_vector(colonies[neighbor_idx], keys)

            # Helical movement: sinusoidal + directional term.
            theta = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0.0, 1.0) * (1.0 - iteration / max(iterations, 1))
            helical = radius * (np.sin(theta) + np.cos(theta)) * rng.normal(size=len(keys))
            directional = rng.uniform(0.0, 1.0) * (neighbor - current)
            elite_pull = rng.uniform(0.0, 0.5) * (current_best - current)

            candidate = current + directional + elite_pull + 0.08 * scale * helical
            candidate = np.clip(candidate, lo, hi)
            candidate_params = _vector_to_params(candidate, keys, search_space)

            cand_score = objective(candidate_params)
            if cand_score > scores[i]:
                new_colonies.append(candidate_params)
                starvation[i] = 0
            else:
                new_colonies.append(colony)
                starvation[i] += 1

            # Starvation adaptation toward best + partial randomization.
            if starvation[i] >= 2 and rng.random() < adaptation_probability:
                adapted = 0.7 * current_best + 0.3 * rng.uniform(lo, hi)
                new_colonies[-1] = _vector_to_params(adapted, keys, search_space)
                starvation[i] = 0

        colonies = new_colonies
        history.append(
            {
                "iteration": iteration,
                "best_score": float(best_score),
                "mean_score": float(scores.mean()),
                "starvation_mean": float(starvation.mean()),
                "best_params": dict(best_params) if best_params else None,
            }
        )

        if no_improve >= patience:
            break

    return SearchResult(
        best_params=best_params or colonies[0],
        best_score=float(best_score),
        history=history,
        runtime_seconds=float(time.perf_counter() - start),
    )


def random_search(
    objective: Callable[[Dict], float], search_space: Dict, budget: int, seed: int = 17
) -> SearchResult:
    rng = np.random.default_rng(seed)
    start = time.perf_counter()
    history = []
    best_score = -np.inf
    best_params = None
    for i in range(budget):
        p = _sample_from_space(rng, search_space)
        s = float(objective(p))
        history.append({"iteration": i, "score": s, "params": p})
        if s > best_score:
            best_score, best_params = s, dict(p)
    return SearchResult(best_params, best_score, history, time.perf_counter() - start)


def pso_search(
    objective: Callable[[Dict], float], search_space: Dict, population: int, iterations: int, seed: int = 17
) -> SearchResult:
    rng = np.random.default_rng(seed)
    keys = list(search_space)
    lo = np.array([search_space[k][0] for k in keys], float)
    hi = np.array([search_space[k][1] for k in keys], float)
    pos = rng.uniform(lo, hi, size=(population, len(keys)))
    vel = np.zeros_like(pos)
    pbest = pos.copy()
    pbest_s = np.full(population, -np.inf)
    gbest = pos[0].copy()
    gbest_s = -np.inf
    history = []
    start = time.perf_counter()

    for t in range(iterations):
        scores = np.zeros(population)
        for i in range(population):
            params = _vector_to_params(pos[i], keys, search_space)
            scores[i] = objective(params)
            if scores[i] > pbest_s[i]:
                pbest_s[i], pbest[i] = scores[i], pos[i].copy()
            if scores[i] > gbest_s:
                gbest_s, gbest = scores[i], pos[i].copy()

        r1, r2 = rng.random(pos.shape), rng.random(pos.shape)
        vel = 0.72 * vel + 1.49 * r1 * (pbest - pos) + 1.49 * r2 * (gbest - pos)
        pos = np.clip(pos + vel, lo, hi)
        history.append({"iteration": t, "best_score": float(gbest_s), "mean_score": float(scores.mean())})

    return SearchResult(
        _vector_to_params(gbest, keys, search_space),
        float(gbest_s),
        history,
        time.perf_counter() - start,
    )


def ga_search(
    objective: Callable[[Dict], float], search_space: Dict, population: int, iterations: int, seed: int = 17
) -> SearchResult:
    rng = np.random.default_rng(seed)
    keys = list(search_space)
    lo = np.array([search_space[k][0] for k in keys], float)
    hi = np.array([search_space[k][1] for k in keys], float)
    pop = rng.uniform(lo, hi, size=(population, len(keys)))
    history = []
    best_vec = pop[0].copy()
    best_score = -np.inf
    start = time.perf_counter()

    for t in range(iterations):
        scores = np.array([objective(_vector_to_params(v, keys, search_space)) for v in pop])
        order = np.argsort(scores)[::-1]
        if scores[order[0]] > best_score:
            best_score = float(scores[order[0]])
            best_vec = pop[order[0]].copy()

        elite_n = max(2, population // 4)
        elites = pop[order[:elite_n]]
        children = [best_vec.copy()]
        while len(children) < population:
            a, b = elites[rng.integers(elite_n)], elites[rng.integers(elite_n)]
            mix = rng.random(len(keys))
            child = mix * a + (1 - mix) * b
            child += rng.normal(0, 0.05, len(keys)) * (hi - lo)
            children.append(np.clip(child, lo, hi))
        pop = np.array(children)
        history.append({"iteration": t, "best_score": best_score, "mean_score": float(scores.mean())})

    return SearchResult(
        _vector_to_params(best_vec, keys, search_space),
        best_score,
        history,
        time.perf_counter() - start,
    )


def gwo_search(
    objective: Callable[[Dict], float], search_space: Dict, population: int, iterations: int, seed: int = 17
) -> SearchResult:
    rng = np.random.default_rng(seed)
    keys = list(search_space)
    lo = np.array([search_space[k][0] for k in keys], float)
    hi = np.array([search_space[k][1] for k in keys], float)
    wolves = rng.uniform(lo, hi, size=(population, len(keys)))
    history = []
    best_score = -np.inf
    start = time.perf_counter()

    for t in range(iterations):
        scores = np.array([objective(_vector_to_params(v, keys, search_space)) for v in wolves])
        order = np.argsort(scores)[::-1]
        alpha, beta, delta = wolves[order[0]], wolves[order[min(1, population-1)]], wolves[order[min(2, population-1)]]
        best_score = max(best_score, float(scores[order[0]]))
        a = 2 - 2 * t / max(iterations - 1, 1)

        new = []
        for x in wolves:
            candidates = []
            for leader in (alpha, beta, delta):
                r1, r2 = rng.random(len(keys)), rng.random(len(keys))
                A = 2 * a * r1 - a
                C = 2 * r2
                D = np.abs(C * leader - x)
                candidates.append(leader - A * D)
            new.append(np.clip(np.mean(candidates, axis=0), lo, hi))
        wolves = np.asarray(new)
        history.append({"iteration": t, "best_score": float(scores[order[0]]), "mean_score": float(scores.mean())})

    scores = np.array([objective(_vector_to_params(v, keys, search_space)) for v in wolves])
    idx = int(np.argmax(scores))
    return SearchResult(
        _vector_to_params(wolves[idx], keys, search_space),
        float(scores[idx]),
        history,
        time.perf_counter() - start,
    )


def bayesian_search(
    objective: Callable[[Dict], float], search_space: Dict, budget: int, seed: int = 17
) -> SearchResult:
    import optuna

    start = time.perf_counter()
    history = []

    def trial_objective(trial):
        p = {}
        for k, (lo, hi) in search_space.items():
            if k in {"attention_dim", "recurrent_hidden", "attention_heads"}:
                p[k] = trial.suggest_int(k, int(lo), int(hi))
            elif k in {"learning_rate", "weight_decay"}:
                p[k] = trial.suggest_float(k, float(lo), float(hi), log=True)
            else:
                p[k] = trial.suggest_float(k, float(lo), float(hi))
        p = repair_hparams(p, search_space)
        s = float(objective(p))
        history.append({"iteration": trial.number, "score": s, "params": p})
        return s

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(trial_objective, n_trials=budget, show_progress_bar=False)
    return SearchResult(
        repair_hparams(study.best_params, search_space),
        float(study.best_value),
        history,
        time.perf_counter() - start,
    )

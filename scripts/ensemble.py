"""Weighted soft-vote ensemble of soces-pubmed checkpoints (v6).

Loads p_include scores from two or three eval-torch JSON files, fits weights
on the validation set by maximising recall subject to precision >= PREC_FLOOR,
then applies those weights to the test set and writes a drop-in eval JSON.

    # with v5 available:
    python scripts/ensemble.py \\
        --checkpoints runs/eval_torch_soces-pubmed-v2.json \\
                      runs/eval_torch_soces-pubmed-v3.json \\
                      runs/eval_torch_soces-pubmed-v5.json \\
        --val  data/val.jsonl \\
        --test data/test_judged.jsonl \\
        --out  runs/eval_torch_ensemble-v6.json

    # with only v2 + v3 (can run now, before v5 exists):
    python scripts/ensemble.py \\
        --checkpoints runs/eval_torch_soces-pubmed-v2.json \\
                      runs/eval_torch_soces-pubmed-v3.json \\
        --val  data/val.jsonl \\
        --test data/test_judged.jsonl \\
        --out  runs/eval_torch_ensemble-v2v3.json

Weights are searched over a uniform simplex grid (GRID_STEPS^(n-1) points).
The objective is max recall at precision >= PREC_FLOOR at the decision threshold
that maximises F1 on the validation set.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

PREC_FLOOR  = 0.28   # minimum acceptable precision on val set
GRID_STEPS  = 25     # simplex grid resolution per weight axis
TAU_GRID    = [t / 100 for t in range(1, 100)]  # thresholds to sweep

# --------------------------------------------------------------------------- #

def load_scores(path: Path) -> dict[int, float]:
    """id -> p_include from an eval_torch JSON."""
    rows = json.loads(path.read_text())["rows"]
    return {r["id"]: float(r["p_include"]) for r in rows}


def load_labels(jsonl_path: Path) -> dict[int, int]:
    """id -> 1 (include) / 0 (exclude) from a judged JSONL."""
    labels = {}
    for line in jsonl_path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        labels[r["id"]] = 1 if r["planted"] == "none" else 0
    return labels


def blend(scores_list: list[dict[int, float]],
          weights: list[float]) -> dict[int, float]:
    """Weighted average of p_include across checkpoints for shared ids."""
    ids = set(scores_list[0])
    for s in scores_list[1:]:
        ids &= set(s)
    return {i: sum(w * s[i] for w, s in zip(weights, scores_list)) for i in ids}


def metrics_at_tau(blended: dict[int, float], labels: dict[int, int],
                   tau: float) -> tuple[float, float, float, int, int, int]:
    """recall, precision, f1, tp, fp, fn at a fixed threshold."""
    tp = fp = fn = 0
    for id_, p in blended.items():
        if id_ not in labels:
            continue
        pred = 1 if p >= tau else 0
        y    = labels[id_]
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 1: fn += 1
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return rec, prec, f1, tp, fp, fn


def best_tau(blended: dict[int, float], labels: dict[int, int],
             prec_floor: float) -> tuple[float, float, float, float]:
    """Return (tau, recall, prec, f1) at the threshold with best F1 subject to prec >= floor."""
    best = (0.5, 0.0, 0.0, 0.0)   # (tau, recall, prec, f1)
    for tau in TAU_GRID:
        rec, prec, f1, *_ = metrics_at_tau(blended, labels, tau)
        if prec >= prec_floor and f1 > best[3]:
            best = (tau, rec, prec, f1)
    return best


def simplex_grid(n: int, steps: int) -> list[list[float]]:
    """All weight vectors on an n-simplex with resolution 1/steps."""
    points = []
    for combo in itertools.combinations_with_replacement(range(steps + 1), n - 1):
        cuts = [0] + list(combo) + [steps]
        w = [cuts[i + 1] - cuts[i] for i in range(n)]
        if sum(w) == steps:
            points.append([x / steps for x in w])
    return points


def search_weights(scores_list: list[dict[int, float]], val_labels: dict[int, int],
                   prec_floor: float, grid_steps: int) -> tuple[list[float], float, float, float, float]:
    """Grid-search weights on val, return (weights, tau, recall, prec, f1).

    Falls back to best-F1 regardless of precision floor if no combination
    achieves the floor (e.g. when val labels are unavailable / all scores too low).
    """
    n      = len(scores_list)
    grid   = simplex_grid(n, grid_steps)
    best        = (None, 0.5, 0.0, 0.0, 0.0)  # respects prec_floor
    best_noflo  = (None, 0.5, 0.0, 0.0, 0.0)  # ignores prec_floor (fallback)

    for weights in grid:
        blended = blend(scores_list, weights)
        tau, rec, prec, f1 = best_tau(blended, val_labels, prec_floor)
        if f1 > best[4]:
            best = (weights, tau, rec, prec, f1)
        # fallback: best F1 ignoring floor
        tau2, rec2, prec2, f12 = best_tau(blended, val_labels, 0.0)
        if f12 > best_noflo[4]:
            best_noflo = (weights, tau2, rec2, prec2, f12)

    if best[0] is not None:
        return best  # type: ignore[return-value]
    # nothing met the precision floor — warn and use fallback
    print(f"  ⚠ no weight combination reached prec_floor={prec_floor:.2f} on val; "
          f"using best-F1 fallback (prec={best_noflo[3]:.2f})", file=__import__("sys").stderr)
    return best_noflo  # type: ignore[return-value]


def full_sweep(blended: dict[int, float], labels: dict[int, int]) -> list[dict]:
    rows_out = []
    for tau in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        rec, prec, f1, tp, fp, fn = metrics_at_tau(blended, labels, tau)
        rows_out.append({"tau": tau, "pred_pos": tp + fp, "tp": tp, "fp": fp, "fn": fn,
                         "recall": round(rec, 4), "precision": round(prec, 4), "f1": round(f1, 4)})
    return rows_out


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="2–3 eval_torch JSON files (order: v2, v3, v5)")
    ap.add_argument("--val",  required=True, help="data/val.jsonl")
    ap.add_argument("--test", required=True, help="data/test_judged.jsonl")
    ap.add_argument("--out",  required=True, help="output eval JSON path")
    ap.add_argument("--prec-floor", type=float, default=PREC_FLOOR,
                    help=f"minimum precision on val (default {PREC_FLOOR})")
    ap.add_argument("--grid-steps", type=int, default=GRID_STEPS,
                    help=f"simplex grid resolution (default {GRID_STEPS})")
    args = ap.parse_args()

    ckpts = [Path(c) for c in args.checkpoints]
    for c in ckpts:
        if not c.exists():
            sys.exit(f"checkpoint not found: {c}")

    scores_list = [load_scores(c) for c in ckpts]
    val_labels  = load_labels(Path(args.val))
    test_labels = load_labels(Path(args.test))

    n_val_pos  = sum(val_labels.values())
    n_test_pos = sum(test_labels.values())
    print(f"val:  {len(val_labels)} items, {n_val_pos} includes")
    print(f"test: {len(test_labels)} items, {n_test_pos} includes")
    print(f"checkpoints: {[c.name for c in ckpts]}")
    print(f"grid: simplex({len(ckpts)}, steps={args.grid_steps})  "
          f"prec_floor={args.prec_floor}")

    weights, val_tau, val_rec, val_prec, val_f1 = search_weights(
        scores_list, val_labels, args.prec_floor, args.grid_steps
    )
    w_str = "  ".join(f"{c.stem.split('-')[-1]}={w:.2f}" for c, w in zip(ckpts, weights))
    print(f"\nOptimal weights:  {w_str}")
    print(f"Val performance:  τ={val_tau:.2f}  recall={val_rec:.4f}  "
          f"prec={val_prec:.4f}  F1={val_f1:.4f}")

    # apply to test set
    test_blended = blend(scores_list, weights)
    tau, rec, prec, f1, tp, fp, fn = (*best_tau(test_blended, test_labels, args.prec_floor),
                                       *[0]*3)
    # recompute tp/fp/fn at the chosen tau
    rec, prec, f1, tp, fp, fn = metrics_at_tau(test_blended, test_labels, val_tau)

    print(f"\nTest performance: τ={val_tau:.2f}  TP={tp}  FP={fp}  FN={fn}")
    print(f"  recall={rec:.4f}  precision={prec:.4f}  F1={f1:.4f}")

    print("\nTest threshold sweep:")
    print(f"{'τ':>5} | {'Pred+':>6} | {'TP':>4} | {'FP':>5} | {'FN':>4} | {'Recall':>7} | {'Prec':>7} | {'F1':>6}")
    print("-" * 62)
    sweep = full_sweep(test_blended, test_labels)
    for row in sweep:
        print(f"{row['tau']:>5.2f} | {row['pred_pos']:>6d} | {row['tp']:>4d} | "
              f"{row['fp']:>5d} | {row['fn']:>4d} | {row['recall']:>7.4f} | "
              f"{row['precision']:>7.4f} | {row['f1']:>6.4f}")

    # save in same format as eval_torch_*.json for drop-in comparison
    out_rows = [{"id": id_, "p_include": round(p, 4),
                 "pick": "include" if p >= val_tau else "exclude",
                 "conf": round(p, 4)}
                for id_, p in sorted(test_blended.items())]
    correct   = sum(1 for r in out_rows if r["pick"] == ("include" if test_labels.get(r["id"]) == 1 else "exclude"))
    out = {
        "ensemble_weights": {c.stem: round(w, 4) for c, w in zip(ckpts, weights)},
        "val_tau": val_tau, "val_recall": round(val_rec, 4),
        "val_precision": round(val_prec, 4), "val_f1": round(val_f1, 4),
        "agreement": round(correct / len(out_rows), 4),
        "recall": round(rec, 4), "precision": round(prec, 4),
        "counts": {"include": sum(1 for r in out_rows if r["pick"] == "include"),
                   "exclude": sum(1 for r in out_rows if r["pick"] == "exclude")},
        "threshold_sweep": sweep,
        "rows": out_rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nsaved → {args.out}")


if __name__ == "__main__":
    main()

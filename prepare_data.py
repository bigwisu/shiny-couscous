"""Data preparation: combine sources, drop test copies, fix the include/exclude mix.

    python prepare_data.py --out data/train_all.jsonl --plot plots/mix_train.png data/train_raw.jsonl

Two stages, each shown in the table and the plot so a skew is caught before training:
1. Drop any row whose SR objective is a near-copy of a test SR objective (guards against
   the same review appearing in both train and test across different export runs).
2. Weight rows to the corpus design (~16.4% include) so the model learns the right prior.
   Weights are not written to the output; they are re-applied at train time by train.py --mix.
"""

import argparse
import collections
import copy
import difflib
import json
import re
from pathlib import Path
from typing import Any

from finetune_data import DESIGN_CLEAN_SHARE, TEST_SET, read_jsonl

NEAR_TEST = 0.8  # SR-objective similarity at which a training row counts as a test copy


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def _test_questions() -> list[str]:
    return [normalise(r["question"]) for r in read_jsonl(TEST_SET)]


def is_near(question: str, tests: list[str], cutoff: float = NEAR_TEST) -> bool:
    q = normalise(question)
    for t in tests:
        sm = difflib.SequenceMatcher(None, q, t)
        if sm.real_quick_ratio() >= cutoff and sm.quick_ratio() >= cutoff and sm.ratio() >= cutoff:
            return True
    return False


def drop_test_copies(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(kept, dropped). Removes rows whose SR objective is a near-copy of a test SR objective."""
    tests = _test_questions()
    near = [is_near(r["question"], tests) for r in rows]
    return [r for r, n in zip(rows, near) if not n], [r for r, n in zip(rows, near) if n]


def weight_to_design(rows: list[dict[str, Any]], clean_share: float = DESIGN_CLEAN_SHARE,
                     balance_types: bool = False) -> None:
    """Weight rows so 'include' has clean_share of total weight.

    balance_types is accepted for API compatibility but is a no-op for the binary task —
    there is only one non-include class.
    """
    n = len(rows)
    counts = collections.Counter(r["planted"] for r in rows)
    for r in rows:
        if r["planted"] == "none":
            r["weight"] = clean_share / (counts["none"] / n)
        else:
            r["weight"] = (1 - clean_share) / ((n - counts["none"]) / n)


def weighted(rows: list[dict[str, Any]], mix: str) -> list[dict[str, Any]]:
    """A weighted copy, leaving the input untouched."""
    out = copy.deepcopy(rows)
    weight_to_design(out)
    return out


def combine(paths: list[str]) -> list[dict[str, Any]]:
    seen: dict[int, dict[str, Any]] = {}
    for path in paths:
        for r in read_jsonl(Path(path)):
            seen.setdefault(r["id"], r)
    return list(seen.values())


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    from mix_report import plot, print_table, problems, shares

    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--plot", required=True)
    args = ap.parse_args()

    raw = combine(args.sources)
    kept, dropped = drop_test_copies(raw)
    stages = [
        ("as generated", raw, False),
        ("after dropping test copies", kept, False),
        ("weighted to design", weighted(kept, "balanced"), True),
    ]
    print_table(stages)
    for name, rows, is_weighted in stages[1:]:
        print(f"{name:35s} {'; '.join(problems(shares(rows, is_weighted))) or 'on design'}")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plot(stages, args.plot)
    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in kept))
    print(f"dropped {len(dropped)} test copies; wrote {len(kept)} rows to {args.out}; plot {args.plot}")


if __name__ == "__main__":
    main()

"""Turn corpus rows into Laya training sequences for citation screening.

Each row has:
    id, question (SR objective), answer (doc title + abstract),
    planted ("none" = include, "excluded" = exclude)

The target is a one-hot on the planted label, built via Laya's own build_sequence so
training sees exactly what inference will see.
"""

import json
import os
from pathlib import Path
from typing import Any

from judge_task import laya_questions

MODEL_ID, SUBFOLDER = "convaiinnovations/laya", "typed-decisions"

# "none" = include (clean / pass); "excluded" = exclude (defective / rework)
DESIGN_CLEAN_SHARE = 0.164   # corpus-wide positive rate: 2526 / 15360

TEST_SET = Path(__file__).parent / "data" / "test_judged.jsonl"


def base_checkpoint() -> str:
    """Local path of the zero-shot checkpoint every result is compared against."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(MODEL_ID, allow_patterns=[f"{SUBFOLDER}/*"])
    path = os.path.join(root, SUBFOLDER)
    return path


def questions(_criterion_labels: str = "short") -> dict[str, dict[str, Any]]:
    """The single screening question in Laya's schema."""
    return laya_questions()


def labels(row: dict[str, Any], _reference: str = "planted") -> dict[str, str]:
    """The answer key for one row (only planted labels exist for this task)."""
    verdict = "include" if row["planted"] == "none" else "exclude"
    return {"verdict": verdict}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# v3: soft label constants — prevent logit saturation on true inclusions
SOFT_INCLUDE_TARGET = 0.90   # confidence ceiling for inclusions  (was 1.0)
SOFT_EXCLUDE_FLOOR  = 0.10   # residual uncertainty for inclusions (was 0.0)


def to_items(rows: list[dict[str, Any]], qs: dict[str, dict[str, Any]], tok, cfg,
             _reference: str = "planted") -> list[dict[str, Any]]:
    """One training sequence per row, dropping any whose markers were cut.

    v3: true-inclusion targets are soft ([0.90, 0.10]) so the model can express
    calibrated intermediate probabilities on borderline evidence instead of being
    driven to logit saturation at P=0 / P=1.  Exclusion targets remain hard [0.0, 1.0].
    """
    from laya.common import QTYPES, build_sequence, render_options
    items = []
    for row in rows:
        state = {"QUESTION": row["question"], "ANSWER": row["answer"]}
        for name, q in qs.items():
            internal = {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}
            keys = list(q["criteria"])
            ids, markers = build_sequence(tok, state, internal, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(internal)):
                continue
            label = keys.index(labels(row)["verdict"])
            if label == 0:  # include → soft target
                target = [SOFT_INCLUDE_TARGET if i == label else SOFT_EXCLUDE_FLOOR
                          for i in range(len(keys))]
            else:           # exclude → hard target (already correct; keep gradient sharp)
                target = [1.0 if i == label else 0.0 for i in range(len(keys))]
            items.append({"ids": ids, "markers": markers, "qtype": QTYPES[q["type"]], "target": target,
                          "label": label, "question": name, "weight": row.get("weight", 1.0),
                          "bge_sim": row.get("bge_sim", 0.0)})
    return items

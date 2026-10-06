"""Fast PyTorch-based GPU evaluation on Linux / CUDA cloud instances.

Runs batched inference across any judged JSONL in ~15-20 seconds on GPU.

    python evaluate_torch.py runs/soces-pubmed-v10
    python evaluate_torch.py base
    python evaluate_torch.py runs/soces-pubmed-v10 --val data/val.jsonl \
        --out runs/eval_val_soces-pubmed-v10.json
"""

import collections
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer
from laya.common import build_model, collate_items, render_options, build_sequence, QTYPES
from huggingface_hub import snapshot_download

from finetune_data import MODEL_ID, SUBFOLDER, TEST_SET, labels, questions, read_jsonl
from judge_task import state_for

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUNS = Path(__file__).parent / "runs"


def load_checkpoint(model_dir: str):
    if model_dir == "base":
        root = snapshot_download(MODEL_ID, allow_patterns=[f"{SUBFOLDER}/*"])
        model_dir = os.path.join(root, SUBFOLDER)
        enc_dir = os.path.join(model_dir, "encoder")
        tok_dir = os.path.join(model_dir, "tokenizer")
    else:
        enc_dir = os.path.join(model_dir, "encoder")
        tok_dir = os.path.join(model_dir, "tokenizer")

    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    model = build_model(cfg, encoder_dir=enc_dir)
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    tok = AutoTokenizer.from_pretrained(tok_dir)
    return model.to(DEVICE).eval(), cfg, tok


@torch.no_grad()
def evaluate_model(model_path: str, out_override: Path | None = None):
    model, cfg, tok = load_checkpoint(model_path)
    items = read_jsonl(TEST_SET)
    qs = questions()

    q_name = list(qs.keys())[0]
    q_spec = qs[q_name]
    internal = {"t": q_spec["type"], "ins": q_spec["instructions"], "crit": q_spec["criteria"]}
    keys = list(q_spec["criteria"])  # ['include', 'exclude']

    rows = []
    t0 = time.time()
    batch_size = 32

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        batch_items = []
        valid_chunk = []
        for it in chunk:
            state = {"QUESTION": it["question"], "ANSWER": it["answer"]}
            ids, markers = build_sequence(tok, state, internal, cfg["max_len"], cfg["head_max_len"])
            if len(markers) != len(render_options(internal)):
                continue
            batch_items.append({
                "ids": ids,
                "markers": markers,
                "qtype": QTYPES[q_spec["type"]],
                "target": [0.0] * len(keys),
                "label": 0,
                "question": q_name,
                "weight": 1.0,
            })
            valid_chunk.append(it)

        if not batch_items:
            continue

        collated = collate_items([batch_items], tok.pad_token_id)
        tensor_keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32):
            logits, _ = model(*(collated[k].to(DEVICE) for k in tensor_keys))
        
        mask = collated["marker_mask"].to(DEVICE)
        probs = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1).cpu().numpy()

        for it, p in zip(valid_chunk, probs):
            p_include = float(p[0])
            pick = "include" if p_include >= 0.50 else "exclude"
            rows.append({
                "id": it["id"],
                "p_include": round(p_include, 4),
                "pick": pick,
                "conf": round(float(max(p)), 4),
            })

    key = {it["id"]: labels(it)["verdict"] for it in items}
    total = len(rows)
    correct = sum(r["pick"] == key[r["id"]] for r in rows)
    agreement = correct / total

    include_ids = {id_ for id_, v in key.items() if v == "include"}
    true_positives = sum(r["pick"] == "include" for r in rows if r["id"] in include_ids)
    recall = true_positives / len(include_ids)

    predicted_includes = sum(r["pick"] == "include" for r in rows)
    precision = true_positives / predicted_includes if predicted_includes > 0 else 0.0

    counts = collections.Counter(r["pick"] for r in rows)
    elapsed = time.time() - t0

    print("=" * 60)
    print(f"EVALUATION RESULTS FOR: {model_path}")
    print(f"Evaluated {total} items in {elapsed:.2f}s ({total/elapsed:.1f} seq/s)")
    print(f"Agreement (Accuracy at P>=0.50): {agreement:.4f} ({correct}/{total})")
    print(f"Recall (Sensitivity at P>=0.50): {recall:.4f} ({true_positives}/{len(include_ids)})")
    print(f"Precision (at P>=0.50):          {precision:.4f} ({true_positives}/{predicted_includes})")
    print(f"Picks Distribution (at P>=0.50): {counts}")
    
    print("\nOperating Threshold Sweep on P(include):")
    print("Thresh | Pred_Inc | TP  | FP   | Recall  | Precision | F1")
    print("-" * 55)
    labels_arr = [1 if key[r["id"]] == "include" else 0 for r in rows]
    probs_arr = [r["p_include"] for r in rows]
    for t in [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]:
        t_pred = [1 if p >= t else 0 for p in probs_arr]
        t_tp = sum(1 for p, y in zip(t_pred, labels_arr) if p == 1 and y == 1)
        t_fp = sum(1 for p, y in zip(t_pred, labels_arr) if p == 1 and y == 0)
        t_fn = sum(1 for p, y in zip(t_pred, labels_arr) if p == 0 and y == 1)
        t_rec = t_tp / (t_tp + t_fn) if (t_tp + t_fn) > 0 else 0
        t_prec = t_tp / (t_tp + t_fp) if (t_tp + t_fp) > 0 else 0
        t_f1 = 2 * t_prec * t_rec / (t_prec + t_rec) if (t_prec + t_rec) > 0 else 0
        print(f"{t:6.2f} | {sum(t_pred):8d} | {t_tp:3d} | {t_fp:4d} | {t_rec:7.4f} | {t_prec:9.4f} | {t_f1:.4f}")
    print("=" * 60)

    tag = "base" if model_path == "base" else Path(model_path).name
    RUNS.mkdir(exist_ok=True)
    out_file = out_override if out_override else RUNS / f"eval_torch_{tag}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({
        "agreement": round(agreement, 4),
        "recall": round(recall, 4),
        "precision": round(precision, 4),
        "counts": dict(counts),
        "rows": rows,
    }, indent=2))
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("model", nargs="?", default="runs/soces-pubmed")
    _p.add_argument("--val",  default=None,
                    help="evaluate on this judged JSONL instead of the default test set")
    _p.add_argument("--out",  default=None,
                    help="write results to this path instead of the default runs/ location")
    _args = _p.parse_args()
    if _args.val:
        TEST_SET = Path(_args.val)
    evaluate_model(_args.model, out_override=Path(_args.out) if _args.out else None)

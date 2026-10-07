"""Produce runs/bioclinical-v0: Laya head weights + BioClinical encoder, no fine-tuning.

Loads the Laya base checkpoint, replaces the encoder backbone with
thomas-sounack/BioClinical-ModernBERT-large, runs temperature calibration on
the same 10 % calib slice used by train.py, and saves the checkpoint.

This is the zero-shot baseline for the BioClinical backbone — no training data
from any of the 66 SRs is used to update weights, only to fit the single
scalar temperature (isotonic calibration, same as every other run).

    python scripts/save_bioclinical_v0.py
    python scripts/save_bioclinical_v0.py --out runs/bioclinical-v0
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from finetune_data import base_checkpoint, questions, read_jsonl, to_items
from calibrate import fit_temperatures_v3 as fit_temperatures
from laya.common import build_model

ENCODER_ID   = "thomas-sounack/BioClinical-ModernBERT-large"
CALIB_SHARE  = 0.1
SEED         = 20260924
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH    = Path(__file__).parent.parent / "data" / "train_all.jsonl"


def forward(model, batch):
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    dtype = torch.bfloat16 if DEVICE.type == "cuda" else torch.float32
    with torch.autocast(DEVICE.type, dtype=dtype):
        logits, act = model(*(batch[k].to(DEVICE) for k in keys))
    return logits.float(), act


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/bioclinical-v0")
    args = ap.parse_args()

    model_dir = base_checkpoint()
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    cfg["encoder"] = ENCODER_ID                            # swap backbone
    model = build_model(cfg, encoder_dir=None)             # download BioClinical from HF
    model.load_state_dict(
        load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = model.to(DEVICE).eval()

    rows = read_jsonl(DATA_PATH)
    random.Random(SEED).shuffle(rows)
    n_calib = int(len(rows) * CALIB_SHARE)
    calib   = to_items(rows[:n_calib], questions(), tok, cfg)
    print(f"{len(rows)} rows → {n_calib} calib sequences for temperature fit", flush=True)

    temps = fit_temperatures(model, forward, calib, tok.pad_token_id)

    out = args.out
    os.makedirs(out, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    tok.save_pretrained(os.path.join(out, "tokenizer"))
    cfg = {**cfg, "fine_tuned": False, "model_name": "bioclinical-v0", "temperature": temps}
    cfg.pop("temperature_by_options", None)
    json.dump(cfg, open(os.path.join(out, "rl_agent_config.json"), "w"), indent=2)
    print(f"saved {out}  temperatures {[round(t, 3) for t in temps]}")


if __name__ == "__main__":
    main()

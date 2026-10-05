"""Fine-tune Laya on citation screening.

    python train.py --train data/train_all.jsonl --out runs/soces-pubmed-v5
    python train.py --train data/train_all.jsonl --max-steps 25   # smoke-test

v5 changes vs v4:
  - loss_on: RLCD dropped entirely. CE-only with pos_weight=5× (back to v2 formula).
  - train(): staged encoder unfreezing — head-only for FREEZE_EPOCHS, then full
    fine-tune.  Prevents bimodal collapse seen in v4 where the encoder shifted into
    exclusion mode before the head had calibrated to the class weights.
  - train(): val monitoring kept (--val), but early-stop trigger removed.
    CE loss is bounded, so automatic stopping is not needed.

Retained from v3/v4:
  - LR_ENCODER=8e-6, LR_HEAD=1e-4 with CosineAnnealingLR.
  - Soft inclusion targets [0.90, 0.10] in finetune_data.py.
  - Hard-negative mining: 1 pos + 1 hard-neg + 2 easy-neg per batch.
  - Isotonic margin calibration T<=2.5 in calibrate.py.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from laya.common import build_model, collate_items

from calibrate import fit_temperatures_v3 as fit_temperatures
from mix_report import preflight
from finetune_data import base_checkpoint, questions, read_jsonl, to_items
from prepare_data import weight_to_design

MICRO_BATCH, GRAD_ACCUM = 4, 16        # effective batch = 64
LR_ENCODER, LR_HEAD = 8.0e-6, 1.0e-4  # conservative encoder LR (v3)
SIGMA_START, SIGMA_END = 0.4, 0.1     # kept for API compat; unused in CE-only loss
CALIB_SHARE, SEED = 0.1, 20260924
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# v5: staged unfreezing — train head-only for this many epochs, then unfreeze encoder
FREEZE_EPOCHS = 2

# hard-negative threshold (TF-IDF cosine sim; top ~10% of exclude scores)
HARD_NEG_THRESHOLD = 0.05

# val monitoring
VAL_RECALL_THRESHOLD = 0.30


def load_model(model_dir: str):
    cfg = json.load(open(os.path.join(model_dir, "rl_agent_config.json")))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    return model.to(DEVICE), cfg, AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))


def forward(model, batch):
    keys = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")
    with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
        logits, act = model(*(batch[k].to(DEVICE) for k in keys))
    return logits.float(), act


def loss_on(model, batch, _sigma: float = 0.0):
    """v5: CE-only with 5× false-negative penalty.

    RLCD is dropped entirely.  The reinforcement signal had high variance at 16%
    positive rate — every weight tried (1×, 3×, 10×) either left recall unchanged
    or destabilised training.  CE with pos_weight=5× is the v2 formula that gave
    the best F1 (0.3440) of any run.

    pos_weight = 5× applies only to true inclusions (target[:, 0] > 0.5).
    Soft labels [0.90, 0.10] from finetune_data.py mean the inclusion slot is 0.90,
    so the > 0.5 threshold still fires correctly.
    """
    logits, act = forward(model, batch)
    mask, target = batch["marker_mask"].to(DEVICE), batch["target"].to(DEVICE)

    is_include = (target[:, 0] > 0.5).float()
    pos_weight  = 1.0 + 4.0 * is_include   # 5.0 for includes, 1.0 for excludes

    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1) * pos_weight
    w  = torch.tensor([m["weight"] for m in batch["meta"]], device=DEVICE)
    return (ce * w).sum() / w.sum() + 0.0 * act.sum()


def build_balanced_batches(items: list[dict], micro_batch_size: int,
                           rng: random.Random) -> list[list[dict]]:
    """1 positive + 1 hard-negative + 2 easy-negatives per micro-batch."""
    pos      = [it for it in items if it["label"] == 0]
    hard_neg = [it for it in items if it["label"] != 0 and it.get("bge_sim", 0.0) >= HARD_NEG_THRESHOLD]
    easy_neg = [it for it in items if it["label"] != 0 and it.get("bge_sim", 0.0) <  HARD_NEG_THRESHOLD]

    if not hard_neg:
        easy_neg = [it for it in items if it["label"] != 0]

    rng.shuffle(pos); rng.shuffle(hard_neg); rng.shuffle(easy_neg)

    batches = []
    n_batches = len(items) // micro_batch_size
    pos_idx = hard_idx = easy_idx = 0

    for _ in range(n_batches):
        batch = []

        # 1 positive
        if pos_idx >= len(pos):
            rng.shuffle(pos); pos_idx = 0
        batch.append(pos[pos_idx]); pos_idx += 1

        # 1 hard negative (fall back to easy if exhausted)
        pool, pidx_attr = (hard_neg, "hard_idx") if hard_neg else (easy_neg, "easy_idx")
        if hard_neg:
            if hard_idx >= len(hard_neg):
                rng.shuffle(hard_neg); hard_idx = 0
            batch.append(hard_neg[hard_idx]); hard_idx += 1
        else:
            if easy_idx >= len(easy_neg):
                rng.shuffle(easy_neg); easy_idx = 0
            batch.append(easy_neg[easy_idx]); easy_idx += 1

        # remaining easy negatives
        for _ in range(micro_batch_size - len(batch)):
            if easy_idx >= len(easy_neg):
                rng.shuffle(easy_neg); easy_idx = 0
            batch.append(easy_neg[easy_idx]); easy_idx += 1

        rng.shuffle(batch)
        batches.append(batch)

    return batches


@torch.no_grad()
def val_recall(model, val_items: list[dict], pad_id: int,
               tau: float = VAL_RECALL_THRESHOLD) -> float:
    """Recall on val set at fixed threshold — for monitoring only, no early-stop."""
    model.eval()
    tp = fn = 0
    for start in range(0, len(val_items), 32):
        chunk    = val_items[start : start + 32]
        collated = collate_items([chunk], pad_id)
        logits, _ = forward(model, collated)
        mask  = collated["marker_mask"].to(DEVICE)
        probs = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1).cpu()
        for it, p in zip(chunk, probs):
            if it["label"] == 0:
                (tp if float(p[0]) >= tau else fn).__class__  # dummy
                if float(p[0]) >= tau: tp += 1
                else:                  fn += 1
    model.train()
    return tp / max(1, tp + fn)


def train(model, items: list[dict], pad_id: int, epochs: int,
          max_steps: int | None, seed: int,
          val_items: list[dict] | None = None) -> None:
    """Staged training: head-only for FREEZE_EPOCHS, then full fine-tune."""
    enc_params  = [p for n, p in model.named_parameters() if     n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]

    # start with encoder frozen
    for p in enc_params:
        p.requires_grad_(False)

    updates = max(1, len(items) // (MICRO_BATCH * GRAD_ACCUM) * epochs)
    # single param group for head only; encoder group added at unfreeze
    opt   = torch.optim.AdamW([{"params": head_params, "lr": LR_HEAD}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)

    model.train()
    step, t0 = 0, time.time()
    encoder_unfrozen = False

    for epoch in range(epochs):

        # --- staged unfreeze ---
        if not encoder_unfrozen and epoch >= FREEZE_EPOCHS:
            for p in enc_params:
                p.requires_grad_(True)
            opt.add_param_group({"params": enc_params, "lr": LR_ENCODER})
            encoder_unfrozen = True
            print(f"epoch {epoch + 1}: encoder unfrozen (LR_enc={LR_ENCODER:.1e})", flush=True)

        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, epochs - 1)
        rng   = random.Random(seed + epoch)
        batches = build_balanced_batches(items, MICRO_BATCH, rng)

        for batch in batches:
            loss = loss_on(model, collate_items([batch], pad_id), sigma) / GRAD_ACCUM
            loss.backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            if step % 25 == 0:
                rate = step * MICRO_BATCH / (time.time() - t0)
                print(f"epoch {epoch + 1}/{epochs} step {step} "
                      f"loss {loss.item() * GRAD_ACCUM:.4f} {rate:.1f} seq/s", flush=True)
            if max_steps and step >= max_steps:
                return

        # val monitoring (no early-stop)
        if val_items:
            rec = val_recall(model, val_items, pad_id, tau=VAL_RECALL_THRESHOLD)
            print(f"epoch {epoch + 1}/{epochs}  val_recall(τ={VAL_RECALL_THRESHOLD:.2f})={rec:.4f}",
                  flush=True)

    print(f"trained {step} steps in {time.time() - t0:.0f} s", flush=True)


def save(model, cfg: dict, tok, temps: list[float], out: str) -> None:
    os.makedirs(out, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(out, "encoder"))
    tok.save_pretrained(os.path.join(out, "tokenizer"))
    cfg = {**cfg, "fine_tuned": True, "model_name": "laya-screening", "temperature": temps}
    cfg.pop("temperature_by_options", None)
    json.dump(cfg, open(os.path.join(out, "rl_agent_config.json"), "w"), indent=2)
    print(f"saved {out}  temperatures (choice, score, noul) {[round(t, 3) for t in temps]}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--rows",  type=int, default=None)
    ap.add_argument("--mix",   choices=("none", "balanced"), default="balanced")
    ap.add_argument("--allow-skew", action="store_true")
    ap.add_argument("--val",   default=None,
                    help="validation JSONL for per-epoch recall monitoring (no early-stop)")
    ap.add_argument("--seed",  type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out",   default=None,
                    help="output directory, e.g. runs/soces-pubmed-v5")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.train))[: args.rows]
    random.Random(args.seed).shuffle(rows)
    if args.mix != "none":
        weight_to_design(rows)
    preflight(rows, args.mix, args.out, args.allow_skew)

    torch.manual_seed(args.seed)
    model, cfg, tok = load_model(base_checkpoint())
    qs = questions()

    n_calib     = int(len(rows) * CALIB_SHARE)
    calib       = to_items(rows[:n_calib], qs, tok, cfg)
    train_items = to_items(rows[n_calib:], qs, tok, cfg)
    print(f"{len(rows)} rows -> {len(train_items)} train / {len(calib)} calib  "
          f"freeze_epochs={FREEZE_EPOCHS}", flush=True)

    val_items = None
    if args.val:
        val_rows  = read_jsonl(Path(args.val))
        val_items = to_items(val_rows, qs, tok, cfg)
        print(f"val: {len(val_rows)} rows -> {len(val_items)} sequences", flush=True)

    train(model, train_items, tok.pad_token_id, args.epochs, args.max_steps,
          args.seed, val_items=val_items)
    if args.out:
        save(model, cfg, tok, fit_temperatures(model, forward, calib, tok.pad_token_id), args.out)


if __name__ == "__main__":
    main()

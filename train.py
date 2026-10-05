"""Fine-tune Laya on citation screening, on one Apple-silicon GPU.

Ported from Convai's official notebook; same loss, learning rates, noise schedule, and
effective batch of 64.  Hardware: one MPS device, bf16 autocast, no DDP.

    python train.py --train data/train_all.jsonl --out runs/screening_v1
    python train.py --train data/train_all.jsonl --max-steps 25   # short smoke-test, saves nothing

Before the first step it prints and plots the class mix (runs/<name>/mix.png) and refuses
to start on a skewed one: see prepare_data.py and mix_report.py.

v4 changes vs v3:
  - loss_on: RLCD asymmetry 3× on loss_rl only; CE is symmetric (no pos_weight on CE term).
  - build_balanced_batches: hard/easy negative partitioning —
      1 positive + 1 hard-negative (bge_sim >= HARD_NEG_THRESHOLD) + 2 easy-negatives per batch.
  - train(): optional per-epoch val-set recall sweep with early-stop (--val flag).
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
from laya.common import build_model, collate_items, proper_reward

from calibrate import fit_temperatures_v3 as fit_temperatures
from mix_report import preflight
from finetune_data import base_checkpoint, questions, read_jsonl, to_items
from prepare_data import weight_to_design

MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = 4, 16, 4  # 4 x 16 = the notebook's 64 per update
LR_ENCODER, LR_HEAD = 8.0e-6, 1.0e-4  # v3: conservative encoder LR to preserve foundation semantics
SIGMA_START, SIGMA_END = 0.4, 0.1
CALIB_SHARE, SEED = 0.1, 20260924
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps")

# v4: hard-negative threshold (TF-IDF cosine sim; top ~10% of exclude scores)
HARD_NEG_THRESHOLD = 0.05
# v4: early-stop if val recall at this threshold drops below floor for N consecutive epochs
VAL_RECALL_THRESHOLD = 0.30
VAL_RECALL_FLOOR = 0.40
VAL_PATIENCE = 2


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


def loss_on(model, batch, sigma: float):
    """v4: RLCD-only asymmetry (3×), CE symmetric.

    Splitting the two signals addresses the v3 failure mode: the 10× dual penalty
    flooded the include bucket by overpowering the exclusion gradient in CE.
    Now:
      - CE is symmetric (no pos_weight) → exclusion boundary stays sharp.
      - RLCD advantage weighted 3× for inclusions → prevents majority-class collapse
        without drowning out the exclusion policy signal.
    """
    logits, act = forward(model, batch)
    mask, target = batch["marker_mask"].to(DEVICE), batch["target"].to(DEVICE)
    k = mask.sum(-1, keepdim=True).float()
    eps = torch.randn((GROUP_SIZE,) + logits.shape, device=DEVICE) * sigma * mask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(DEVICE), mask, w_sph=0.75, w_rps=1.0)
        adv = (r - r.mean(0, keepdim=True)) / ((r - r.mean(0, keepdim=True)).std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma**2)

    # CE: symmetric — no pos_weight, exclusion gradient preserved
    ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1)

    w = torch.tensor([m["weight"] for m in batch["meta"]], device=DEVICE)
    # RLCD: 3× asymmetric advantage weight on inclusions only
    is_include = (target[:, 0] > 0.5).float()
    rl_weight = 1.0 + 2.0 * is_include  # 3.0 for includes, 1.0 for excludes
    loss_rl = -((adv * logp * rl_weight.unsqueeze(0)).mean(0) * w).sum() / w.sum()
    return loss_rl + (ce * w).sum() / w.sum() + 0.0 * act.sum()


def build_balanced_batches(items: list[dict], micro_batch_size: int,
                           rng: random.Random) -> list[list[dict]]:
    """v4: hard/easy negative partitioning.

    Per micro-batch (size 4):
      1 positive  (include)
      1 hard negative  (bge_sim >= HARD_NEG_THRESHOLD — topically similar PICO mismatch)
      2 easy negatives (bge_sim <  HARD_NEG_THRESHOLD — genuinely off-topic)

    If hard negatives run out they are replaced from easy negatives; if positives run out
    they wrap around (same behaviour as before).
    """
    pos  = [it for it in items if it["label"] == 0]
    hard_neg = [it for it in items if it["label"] != 0 and it.get("bge_sim", 0.0) >= HARD_NEG_THRESHOLD]
    easy_neg = [it for it in items if it["label"] != 0 and it.get("bge_sim", 0.0) <  HARD_NEG_THRESHOLD]

    # Fallback: if no hard negatives annotated, treat all negatives as easy
    if not hard_neg:
        easy_neg = [it for it in items if it["label"] != 0]

    rng.shuffle(pos)
    rng.shuffle(hard_neg)
    rng.shuffle(easy_neg)

    batches = []
    n_batches = len(items) // micro_batch_size
    pos_idx = hard_idx = easy_idx = 0

    for _ in range(n_batches):
        batch = []

        # --- 1 positive ---
        if pos_idx >= len(pos):
            rng.shuffle(pos)
            pos_idx = 0
        batch.append(pos[pos_idx])
        pos_idx += 1

        # --- 1 hard negative ---
        if hard_neg:
            if hard_idx >= len(hard_neg):
                rng.shuffle(hard_neg)
                hard_idx = 0
            batch.append(hard_neg[hard_idx])
            hard_idx += 1
        else:
            # no hard negatives — take an extra easy negative instead
            if easy_idx >= len(easy_neg):
                rng.shuffle(easy_neg)
                easy_idx = 0
            batch.append(easy_neg[easy_idx])
            easy_idx += 1

        # --- remaining slots filled with easy negatives ---
        needed = micro_batch_size - len(batch)
        for _ in range(needed):
            if easy_idx >= len(easy_neg):
                rng.shuffle(easy_neg)
                easy_idx = 0
            batch.append(easy_neg[easy_idx])
            easy_idx += 1

        rng.shuffle(batch)
        batches.append(batch)

    return batches


@torch.no_grad()
def val_recall(model, val_items: list[dict], pad_id: int,
               tau: float = VAL_RECALL_THRESHOLD) -> float:
    """Quick recall check on the validation set at a fixed threshold."""
    model.eval()
    tp = fn = 0
    for start in range(0, len(val_items), 32):
        chunk = val_items[start : start + 32]
        collated = collate_items([chunk], pad_id)
        logits, _ = forward(model, collated)
        mask = collated["marker_mask"].to(DEVICE)
        probs = torch.softmax(logits.masked_fill(~mask, -1e4), dim=-1).cpu()
        for it, p in zip(chunk, probs):
            if it["label"] == 0:  # true include
                if float(p[0]) >= tau:
                    tp += 1
                else:
                    fn += 1
    model.train()
    return tp / max(1, tp + fn)


def train(model, items, pad_id: int, epochs: int, max_steps: int | None, seed: int,
          val_items: list[dict] | None = None) -> None:
    enc  = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt  = torch.optim.AdamW([{"params": enc, "lr": LR_ENCODER}, {"params": head, "lr": LR_HEAD}],
                             weight_decay=0.01)
    updates = max(1, len(items) // (MICRO_BATCH * GRAD_ACCUM) * epochs)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=updates, eta_min=1e-6)
    model.train()
    step, t0 = 0, time.time()
    bad_epochs = 0  # consecutive epochs below VAL_RECALL_FLOOR

    for epoch in range(epochs):
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, epochs - 1)
        rng = random.Random(seed + epoch)
        batches = build_balanced_batches(items, MICRO_BATCH, rng)
        for batch in batches:
            loss = loss_on(model, collate_items([batch], pad_id), sigma) / GRAD_ACCUM
            loss.backward()
            step += 1
            if step % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(), sched.step(), opt.zero_grad(set_to_none=True)
            if step % 25 == 0:
                rate = step * MICRO_BATCH / (time.time() - t0)
                print(f"epoch {epoch + 1}/{epochs} step {step} loss {loss.item() * GRAD_ACCUM:.4f} "
                      f"{rate:.1f} seq/s", flush=True)
            if max_steps and step >= max_steps:
                return

        # --- per-epoch val recall sweep ---
        if val_items:
            rec = val_recall(model, val_items, pad_id, tau=VAL_RECALL_THRESHOLD)
            print(f"epoch {epoch + 1}/{epochs}  val_recall(τ={VAL_RECALL_THRESHOLD:.2f})={rec:.4f}",
                  flush=True)
            if rec < VAL_RECALL_FLOOR:
                bad_epochs += 1
                print(f"  ⚠ val recall {rec:.4f} < floor {VAL_RECALL_FLOOR}  "
                      f"({bad_epochs}/{VAL_PATIENCE})", flush=True)
                if bad_epochs >= VAL_PATIENCE:
                    print("  early-stop triggered", flush=True)
                    return
            else:
                bad_epochs = 0  # reset counter on recovery

    print(f"trained {step} steps in {time.time() - t0:.0f} s", flush=True)


def save(model, cfg: dict, tok, temps: list[float], out: str) -> None:
    """Same layout as the published checkpoints so laya and laya-mlx both load it."""
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
    ap.add_argument("--train", required=True, help="prepared file from prepare_data.py")
    ap.add_argument("--rows", type=int, default=None, help="use only the first N rows")
    ap.add_argument("--mix", choices=("none", "balanced"), default="balanced",
                    help="balanced: weight include/exclude to the corpus design (~16.4%% include)")
    ap.add_argument("--allow-skew", action="store_true",
                    help="train even if the mix is off the design")
    ap.add_argument("--val", default=None,
                    help="validation JSONL for per-epoch recall monitoring + early-stop "
                         "(e.g. data/val.jsonl)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None,
                    help="output directory; e.g. runs/soces-pubmed-v4")
    args = ap.parse_args()

    rows = read_jsonl(Path(args.train))[: args.rows]
    random.Random(args.seed).shuffle(rows)
    if args.mix != "none":
        weight_to_design(rows)
    preflight(rows, args.mix, args.out, args.allow_skew)

    torch.manual_seed(args.seed)
    model, cfg, tok = load_model(base_checkpoint())
    qs = questions()

    n_calib = int(len(rows) * CALIB_SHARE)
    calib       = to_items(rows[:n_calib],  qs, tok, cfg)
    train_items = to_items(rows[n_calib:],  qs, tok, cfg)
    print(f"{len(rows)} rows -> {len(train_items)} train / {len(calib)} calibration sequences",
          flush=True)

    val_items = None
    if args.val:
        val_rows  = read_jsonl(Path(args.val))
        val_items = to_items(val_rows, qs, tok, cfg)
        print(f"val: {len(val_rows)} rows -> {len(val_items)} sequences  "
              f"(early-stop τ={VAL_RECALL_THRESHOLD} floor={VAL_RECALL_FLOOR} patience={VAL_PATIENCE})",
              flush=True)

    train(model, train_items, tok.pad_token_id, args.epochs, args.max_steps, args.seed,
          val_items=val_items)
    if args.out:
        save(model, cfg, tok, fit_temperatures(model, forward, calib, tok.pad_token_id), args.out)


if __name__ == "__main__":
    main()

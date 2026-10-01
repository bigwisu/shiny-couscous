# Laya citation-screening fine-tune

Fine-tunes [Laya](https://huggingface.co/convaiinnovations/laya) (Convai, 421 M parameters,
Apache-2.0) to screen PubMed abstracts for systematic reviews: **include** or **exclude**.

## Data source

Training data comes from a curated corpus of systematic reviews indexed in PubMed.
Each systematic review (SR) is identified by its **PubMed ID (`sr_pmid`)**.  For every
SR, a set of candidate PubMed documents (`doc_pmid`) has been labelled by human reviewers
as either included (`is_inclusion = true`) or excluded (`is_inclusion = false`) during
the title-and-abstract screening stage.

| Table | Contents |
|---|---|
| `systematic_reviews` | 70 SRs — title, abstract (objective + inclusion criteria), query embedding |
| `pubmed_documents` | 15 530 candidate documents — title, abstract, dense embedding |
| `sr_document_mappings` | 16 598 SR ↔ document pairs with the inclusion label |

Four SRs (`32199484`, `29187358`, `28348110`, `28903922`) were identified as malformed and
excluded before splitting, leaving **66 usable SRs and 15 360 labelled pairs**.

### Split — frozen by SR group

The split is assigned by `sr_pmid` so that every document from the same review lands
entirely in one partition (splitting by row would leak the review objective into both
train and test).  Groups were sorted by size and assigned round-robin to keep the
include rate (~16 %) consistent across all three partitions.

| File | SR groups | Rows | Positives (include) |
|---|---|---|---|
| `data/train_raw.jsonl` | 22 | 5 751 | 951 (16.5 %) |
| `data/val.jsonl` | 22 | 4 989 | 900 (18.0 %) |
| `data/test_judged.jsonl` | 22 | 4 620 | 675 (14.6 %) |

`data/test_judged.jsonl` is the **sealed held-out set** — do not inspect it until after
training is complete.

Each row is a JSON object:

```json
{
  "id": 0,
  "sr_pmid": "31585960",
  "doc_pmid": "12345678",
  "question": "<SR abstract — objective and inclusion criteria>",
  "answer":   "<doc title>\n\n<doc abstract>",
  "planted":  "none"
}
```

`planted` is `"none"` for included documents (positive / pass) and `"excluded"` for
excluded ones (negative / rework) — matching the convention used by Laya's training
pipeline.

## Project files

| File | Purpose |
|---|---|
| `build_dataset.py` | Export train / val / test JSONL from Postgres *(needs database access)* |
| `generate_dataset.py` | Original small CSV sampler *(needs database access)* |
| `judge_task.py` | Screening task definition — `SCREENING_OPTIONS`, `laya_questions`, `state_for` |
| `finetune_data.py` | Rows → Laya training sequences via `build_sequence` |
| `mix_report.py` | Include / exclude balance table, plot, and pre-flight check |
| `prepare_data.py` | Drop near-test SR copies, weight rows to the design |
| `calibrate.py` | Per-type softmax temperature fit (LBFGS) |
| `train.py` | Fine-tuning loop — MPS device, bf16, RLCD loss |
| `evaluate.py` | Score a checkpoint on `test_judged.jsonl` |

## Training on an Apple M4 Mac

### Requirements

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All steps run on the **MPS** device (Apple silicon GPU) automatically — no config change needed.

### Step 1 — Prepare training data ✅

Drops any training SR whose SR PMID appears in the test set, then shows the
include / exclude mix at each stage and plots it to `plots/mix_train.png`.

```bash
python prepare_data.py \
    --out data/train_all.jsonl \
    --plot plots/mix_train.png \
    data/train_raw.jsonl
```

`train.py` will refuse to start if the mix is off the design; fix it by re-running with
the default `--mix balanced` flag (already the default).

> **Done** — `data/train_all.jsonl` written (5 751 rows, 0 test copies dropped).

### Step 2 — Train (~2–3 h on M4) ⏳

> **Not started.** `MICRO_BATCH` reduced from 8 → 4 (`GRAD_ACCUM` 8 → 16) to fit
> in 20 GB MPS memory; effective batch size unchanged at 64.

```bash
python train.py \
    --train data/train_all.jsonl \
    --mix balanced \
    --out runs/screening_v1
```

Before the first gradient step `train.py` prints and plots the mix it will train on
(`runs/screening_v1/mix.png`) and aborts on a skewed one.  Pass `--allow-skew` to
override, or `--max-steps 25` for a quick smoke-test that saves nothing.

### Step 3 — Evaluate ⏳

```bash
python evaluate.py base                # zero-shot baseline
python evaluate.py runs/screening_v1  # fine-tuned
```

Results are written to `runs/eval_base.json` and `runs/eval_screening_v1.json`.
The output reports **overall agreement** and **recall on includes** — the recall metric
matters more for screening because missing a relevant paper is a worse error than
keeping an irrelevant one.

## Storage requirements

| Item | Size |
|---|---|
| `data/` (three JSONL files + CSV) | ~70 MB |
| Laya base checkpoint (HuggingFace cache) | ~840 MB |
| `runs/screening_v1` (fine-tuned checkpoint) | ~840 MB |
| `plots/`, logs | < 1 MB |
| **Total** | **~1.75 GB** |

Python venv adds ~200–400 MB depending on whether PyTorch was already cached.
Allow **3 GB free** to be comfortable.

## HuggingFace token

Setting a HuggingFace token speeds up model downloads (higher rate limits, gated
model access) and avoids throttling on the ~840 MB Laya checkpoint.

1. Create a read token at <https://huggingface.co/settings/tokens>.
2. Add it to your `.env`:

```
HF_TOKEN=hf_...
```

`train.py` and `evaluate.py` pick it up automatically via the `huggingface_hub`
library.  Alternatively, log in once with the CLI — the token is then stored in
`~/.cache/huggingface/token` and does not need to be in `.env`:

```bash
huggingface-cli login
```

## Database access

`build_dataset.py` and `generate_dataset.py` require a connection to the Postgres
database configured via `.env` (copy `.env.example` and fill in your credentials).
All other scripts work entirely from the JSONL files in `data/` and can be run
without any database access — e.g. from an office machine where the database is
not reachable.

```
HF_TOKEN=hf_...
PG_HOST=
PG_PORT=
PG_USER=
PG_PASSWORD=
PG_DATABASE=
```

The fine-tuned checkpoint is not distributed.  Train your own with Step 2 above;
the evaluation results it produces are saved to `runs/`.

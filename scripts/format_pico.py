"""Reformat the `question` field in training/val/test JSONL files with explicit
PICO structure, using an LLM to extract slots from each SR's objective blob.

Usage:
    # dry-run: print extracted PICO blocks for all 22 SRs, no file writes
    python scripts/format_pico.py --dry-run

    # rewrite train + val + test in-place (originals saved as .pre-pico.bak)
    python scripts/format_pico.py \
        --inputs data/train_all.jsonl data/val.jsonl data/test_judged.jsonl

    # rewrite only one file
    python scripts/format_pico.py --inputs data/train_all.jsonl

How it works:
    1. Collect the unique SR `question` blobs (22 in total across all splits).
    2. For each SR, call openrouter/gpt-6-luna to extract Population, Intervention /
       Topic, Outcome, and Study design into 4–5 short phrases.
    3. Build a structured PICO block (the new `question` value).
    4. Replace `question` in every row that belongs to that SR and write the file.

The LLM is called once per unique SR (22 calls total, ~$0.01).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
load_dotenv(Path(__file__).parent.parent / "soces-pubmed" / ".env")  # fallback
load_dotenv()

MODEL      = "openai/gpt-6-luna"
OR_URL     = "https://openrouter.ai/api/v1/chat/completions"
OR_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

SYSTEM_PROMPT = """\
You are a systematic-review methodologist. Given the objective / background text
of a systematic review, extract PICO-style eligibility criteria in JSON.

Return ONLY valid JSON with exactly these keys:
  population   – who/what is studied (patient group, condition, or exposure)
  intervention – treatment, exposure, test, or topic being evaluated
  outcome      – primary endpoints or effects of interest
  study_design – eligible study types (e.g. RCT, cohort, observational, any)
  date_range   – search date range if mentioned, otherwise null

Keep each value to one or two short phrases. Do not include search database names.
Do not include the word "systematic review" in any value.
"""

PICO_TEMPLATE = """\
[REVIEW CRITERIA]
Population:   {population}
Intervention: {intervention}
Outcome:      {outcome}
Study design: {study_design}{date_line}

[DECISION]
Include if: empirical data on {population};
            relevant to {intervention}; reports or measures {outcome}.
Exclude if: commentary, editorial, animal study, no original data,
            or unrelated population/condition.\
"""


# --------------------------------------------------------------------------- #

def call_llm(question_blob: str, retries: int = 3) -> dict:
    """Return extracted PICO slots as a dict."""
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question_blob[:2000]},
        ],
        "temperature": 0.0,
        "max_tokens": 300,
    }
    headers = {
        "Authorization": f"Bearer {OR_API_KEY}",
        "Content-Type":  "application/json",
    }
    for attempt in range(retries):
        try:
            r = requests.post(OR_URL, json=payload, headers=headers, timeout=30)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            if content is None:
                raise ValueError("LLM returned null content")
            text = content.strip()
            # strip markdown fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)
        except (requests.RequestException, json.JSONDecodeError, KeyError, ValueError) as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"LLM call failed after {retries} attempts: {exc}") from exc


def build_pico_question(slots: dict) -> str:
    date_line = (f"\nDate range:   {slots['date_range']}"
                 if slots.get("date_range") else "")
    return PICO_TEMPLATE.format(
        population   = slots.get("population",   "not specified"),
        intervention = slots.get("intervention", "not specified"),
        outcome      = slots.get("outcome",      "not specified"),
        study_design = slots.get("study_design", "any"),
        date_line    = date_line,
    )


# --------------------------------------------------------------------------- #

def collect_sr_questions(paths: list[Path]) -> dict[str, str]:
    """sr_pmid -> first question blob seen (they are identical within an SR)."""
    mapping: dict[str, str] = {}
    for p in paths:
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            pmid = r.get("sr_pmid", r.get("sr_id", "unknown"))
            if pmid not in mapping:
                mapping[pmid] = r["question"]
    return mapping


def extract_all_pico(sr_questions: dict[str, str],
                     verbose: bool = True) -> dict[str, dict]:
    """Return sr_pmid -> raw slots dict (cache stores slots, not rendered text)."""
    slots_map: dict[str, dict] = {}
    total = len(sr_questions)
    for i, (pmid, blob) in enumerate(sr_questions.items(), 1):
        if verbose:
            print(f"  [{i}/{total}] SR {pmid} ... ", end="", flush=True)
        slots = call_llm(blob)
        slots_map[pmid] = slots
        if verbose:
            pop = slots.get("population", "?")[:60]
            print(f"ok  ({pop}...)")
    return slots_map


def rewrite_jsonl(path: Path, pico_map: dict[str, str]) -> int:
    """Overwrite path with PICO-formatted questions. Returns rows rewritten."""
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    out, changed = [], 0
    for line in lines:
        r = json.loads(line)
        pmid = r.get("sr_pmid", r.get("sr_id", "unknown"))
        if pmid in pico_map:
            r["question"] = pico_map[pmid]
            changed += 1
        out.append(json.dumps(r, ensure_ascii=False))

    # backup original
    bak = path.with_suffix(".pre-pico.bak")
    if not bak.exists():
        path.rename(bak)
    else:
        path.unlink()
    path.write_text("\n".join(out) + "\n")
    return changed


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reformat SR questions in JSONL files with PICO structure."
    )
    ap.add_argument(
        "--inputs", nargs="+",
        default=["data/train_all.jsonl", "data/val.jsonl", "data/test_judged.jsonl"],
        help="JSONL files to reformat (default: all three data files)",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print extracted PICO blocks and exit without writing files",
    )
    ap.add_argument(
        "--cache", default="scripts/pico_cache.json",
        help="JSON file to cache extracted slots (avoids re-calling the LLM)",
    )
    args = ap.parse_args()

    if not OR_API_KEY:
        sys.exit("OPENROUTER_API_KEY not set. Add it to .env or export it.")

    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            sys.exit(f"File not found: {p}")

    # load cache  (stores raw slot dicts, not rendered strings)
    cache_path = Path(args.cache)
    slots_cache: dict[str, dict] = {}
    if cache_path.exists():
        raw = json.loads(cache_path.read_text())
        # upgrade: old cache may have stored rendered strings — discard and re-extract
        slots_cache = {k: v for k, v in raw.items() if isinstance(v, dict)}
        if len(slots_cache) < len(raw):
            print(f"  ⚠ Discarded {len(raw) - len(slots_cache)} stale string entries from cache.")
        print(f"Loaded {len(slots_cache)} cached slot dicts from {cache_path}")

    # collect SRs not yet in cache
    sr_questions = collect_sr_questions(paths)
    missing = {pmid: q for pmid, q in sr_questions.items() if pmid not in slots_cache}
    print(f"Total unique SRs: {len(sr_questions)}  "
          f"(cached: {len(sr_questions) - len(missing)}, to extract: {len(missing)})")

    if missing:
        print(f"Calling {MODEL} for {len(missing)} SR(s)...")
        new_slots = extract_all_pico(missing)
        slots_cache.update(new_slots)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(slots_cache, indent=2, ensure_ascii=False))
        print(f"Cache saved → {cache_path}")

    # render slots → PICO question strings
    pico_map: dict[str, str] = {
        pmid: build_pico_question(slots) for pmid, slots in slots_cache.items()
    }

    if args.dry_run:
        print("\n=== DRY RUN — PICO blocks ===\n")
        for pmid, pico_q in pico_map.items():
            print(f"--- SR {pmid} ---")
            print(pico_q)
            print()
        return

    # rewrite files
    for path in paths:
        n = rewrite_jsonl(path, pico_map)
        print(f"Rewrote {n:5d} rows in {path}  (backup: {path.with_suffix('.pre-pico.bak')})")

    print("\nDone. Run train.py with the updated data files to train v7.")


if __name__ == "__main__":
    main()

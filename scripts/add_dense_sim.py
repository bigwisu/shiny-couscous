"""Annotate training/validation JSONL files with per-row cosine similarity scores.

Adds a ``bge_sim`` field (float 0–1) to every row representing the TF-IDF cosine
similarity between the SR query (``question`` field) and the candidate document
(``answer`` field).  This field drives hard-negative partitioning in
``build_balanced_batches``: rows whose SR is topically similar to the document but
labelled ``excluded`` are PICO mismatches — the hardest negatives to discriminate.

    python scripts/add_dense_sim.py --input data/train_all.jsonl
    python scripts/add_dense_sim.py --input data/val.jsonl

The file is updated in-place (a .bak is kept).  Re-running is idempotent: rows that
already have ``bge_sim`` are re-scored (scores are cheap to recompute).

No GPU and no sentence-transformers required.  TF-IDF cosine is a reliable proxy for
topical overlap at the SR/document level; the absolute threshold (default 0.30) is
intentionally low because we are partitioning on *relative* similarity rank within
each SR, not absolute semantic similarity.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def add_sim(rows: list[dict], threshold: float) -> tuple[list[dict], dict]:
    """Fit one TF-IDF vectoriser per SR, compute query–document cosine similarity."""
    from collections import defaultdict

    # Group rows by SR query
    by_sr: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_sr[r["question"]].append(i)

    hard, easy, total_neg = 0, 0, 0
    for question, idxs in by_sr.items():
        sr_rows = [rows[i] for i in idxs]
        docs = [r["answer"] for r in sr_rows]
        corpus = [question] + docs

        vec = TfidfVectorizer(
            sublinear_tf=True,
            max_features=30_000,
            ngram_range=(1, 2),
            min_df=1,
            dtype=np.float32,
        )
        try:
            tfidf = vec.fit_transform(corpus)
        except ValueError:
            # empty vocabulary (very short texts) — assign 0
            for i in idxs:
                rows[i]["bge_sim"] = 0.0
            continue

        q_vec = tfidf[0]
        d_vecs = tfidf[1:]
        sims = cosine_similarity(q_vec, d_vecs)[0]  # shape (n_docs,)

        for idx, sim in zip(idxs, sims):
            rows[idx]["bge_sim"] = round(float(sim), 4)
            if rows[idx]["planted"] != "none":  # exclude rows
                total_neg += 1
                if sim >= threshold:
                    hard += 1
                else:
                    easy += 1

    stats = {
        "total_neg": total_neg,
        "hard_neg": hard,
        "easy_neg": easy,
        "hard_pct": round(100 * hard / max(1, total_neg), 1),
    }
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate JSONL rows with TF-IDF cosine sim.")
    ap.add_argument("--input", required=True, help="path to train_all.jsonl or val.jsonl")
    ap.add_argument(
        "--threshold", type=float, default=0.05,
        help="cosine sim threshold above which an exclude row is 'hard negative' "
             "(default 0.05 ≈ top-10%% of TF-IDF exclude scores in the training corpus)",
    )
    args = ap.parse_args()

    path = Path(args.input)
    bak = path.with_suffix(".jsonl.bak")
    shutil.copy2(path, bak)
    print(f"backed up {path} → {bak}")

    rows = load_jsonl(path)
    print(f"loaded {len(rows)} rows from {path}")

    rows, stats = add_sim(rows, args.threshold)

    save_jsonl(path, rows)
    print(
        f"annotated {len(rows)} rows  "
        f"hard-neg: {stats['hard_neg']}/{stats['total_neg']} ({stats['hard_pct']}%)  "
        f"threshold={args.threshold}"
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()

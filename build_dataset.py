"""Export train / val / test JSONL files from Postgres for Laya fine-tuning.

Split is frozen by SR group (sr_pmid).  The four malformed reviews are excluded.
Run from the repo root:

    python build_dataset.py

Outputs (relative to repo root):
    examples/laya-finetuning/data/train_raw.jsonl   (~5 751 rows, 22 SR groups)
    examples/laya-finetuning/data/val.jsonl          (~4 989 rows, 22 SR groups)
    examples/laya-finetuning/data/test_judged.jsonl  (~4 620 rows, 22 SR groups)

Each line is a JSON object with the keys expected by finetune_data.py:
    id, question, answer, planted   (plus sr_pmid, doc_pmid for traceability)
"""

import json
import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Frozen SR split — do NOT re-derive; changing this breaks reproducibility.
# ---------------------------------------------------------------------------
EXCLUDED = {"32199484", "29187358", "28348110", "28903922"}  # malformed

TRAIN_SRS = {
    "31585960", "27737830", "26903336", "30917990", "30158148",
    "24727842", "22986378", "31255301", "26868137", "33472813",
    "23814120", "31990319", "30617123", "26199070", "26830055",
    "25059938", "24046285", "32909814", "23529983", "25770113",
    "23935058", "26349907",
}

VAL_SRS = {
    "30326495", "27802478", "28114600", "33148618", "31727627",
    "30383109", "22226047", "26830221", "27548070", "23420235",
    "30409774", "32496521", "27802505", "33441384", "26109551",
    "23033409", "31591158", "25556126", "22872710", "33186535",
    "22323502", "33176180",
}

TEST_SRS = {
    "31200992", "30884526", "27893131", "24157497", "26420598",
    "25569206", "32371466", "32479176", "22777524", "25006006",
    "23460092", "29049756", "32442035", "32427305", "24922745",
    "24592495", "29540345", "27142267", "32459529", "23900314",
    "22422870", "26420387",
}

# "planted" mirrors the laya-finetuning pipeline convention:
#   is_inclusion=True  -> "none"      (clean / pass)
#   is_inclusion=False -> "excluded"  (defective / rework)
PLANTED = {True: "none", False: "excluded"}

QUERY = """
SELECT
    m.sr_pmid,
    m.pmid                  AS doc_pmid,
    sr.abstract             AS sr_objective,
    d.title                 AS doc_title,
    d.abstract              AS doc_abstract,
    m.is_inclusion
FROM sr_document_mappings m
JOIN systematic_reviews  sr ON sr.sr_pmid = m.sr_pmid
JOIN pubmed_documents     d  ON d.pmid    = m.pmid
WHERE m.sr_pmid NOT IN %(excluded)s
ORDER BY m.sr_pmid, m.pmid
"""


def _connect():
    load_dotenv()
    return psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=int(os.environ.get("PG_PORT", 5432)),
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
        dbname=os.environ["PG_DATABASE"],
    )


def _to_row(seq_id: int, rec) -> dict:
    sr_pmid, doc_pmid, sr_objective, doc_title, doc_abstract, is_inclusion = rec
    answer = f"{doc_title}\n\n{doc_abstract}" if doc_title else doc_abstract
    return {
        "id": seq_id,
        "sr_pmid": sr_pmid,
        "doc_pmid": doc_pmid,
        "question": sr_objective,
        "answer": answer,
        "planted": PLANTED[is_inclusion],
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    conn = _connect()
    cur = conn.cursor()
    cur.execute(QUERY, {"excluded": tuple(EXCLUDED)})
    records = cur.fetchall()
    conn.close()

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    split_map = {**{s: "train" for s in TRAIN_SRS},
                 **{s: "val"   for s in VAL_SRS},
                 **{s: "test"  for s in TEST_SRS}}

    for seq_id, rec in enumerate(records):
        sr_pmid = rec[0]
        split = split_map.get(sr_pmid)
        if split is None:
            continue  # safety: unknown group, skip
        splits[split].append(_to_row(seq_id, rec))

    out_dir = Path("examples/laya-finetuning/data")
    targets = {
        "train": out_dir / "train_raw.jsonl",
        "val":   out_dir / "val.jsonl",
        "test":  out_dir / "test_judged.jsonl",
    }
    for name, rows in splits.items():
        _write(targets[name], rows)
        pos = sum(1 for r in rows if r["planted"] == "none")
        neg = len(rows) - pos
        print(f"{name:5s}: {len(rows):5d} rows | {pos:4d} pos ({pos/len(rows):.1%}) "
              f"| {neg:4d} neg | -> {targets[name]}")


if __name__ == "__main__":
    main()

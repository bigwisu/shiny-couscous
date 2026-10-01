"""The citation-screening task both engines answer, defined once so comparisons are like-for-like.

One binary question: given a systematic-review objective (QUESTION) and a candidate abstract
(ANSWER), decide include or exclude.
"""

from typing import Any

SCREENING_OPTIONS: dict[str, str] = {
    "include": "the abstract meets the review's inclusion criteria and should be retrieved for full-text screening",
    "exclude": "the abstract does not meet the inclusion criteria and should be discarded",
}


def laya_questions() -> dict[str, dict[str, Any]]:
    """The screening question in Laya's schema."""
    return {
        "verdict": {
            "type": "choice",
            "instructions": (
                "You are screening citations for a systematic review. "
                "Read the review QUESTION (objective and inclusion criteria) "
                "and the candidate ANSWER (title + abstract). "
                "Decide whether this paper should be included for full-text review."
            ),
            "criteria": SCREENING_OPTIONS,
        }
    }


def state_for(question: str, answer: str) -> dict[str, str]:
    """The item under judgement — identical for both engines."""
    return {"QUESTION": question, "ANSWER": answer}

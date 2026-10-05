"""The citation-screening task both engines answer, defined once so comparisons are like-for-like.

One binary question: given a systematic-review objective (QUESTION) and a candidate abstract
(ANSWER), decide include or exclude.
"""

from typing import Any

SCREENING_OPTIONS: dict[str, str] = {
    "include": "the paper reports empirical data on the topic or population described in the review objective, even indirectly",
    "exclude": "the paper's topic is entirely unrelated to the review objective, or it contains no data (e.g. editorial, commentary, opinion piece)",
}


def laya_questions() -> dict[str, dict[str, Any]]:
    """The screening question in Laya's schema, aligned with the V7 rubric."""
    return {
        "verdict": {
            "type": "choice",
            "instructions": (
                "Given the systematic review objective in QUESTION, does this paper in ANSWER contribute relevant evidence — even indirectly? "
                "Include if the paper reports empirical data on the topic or population described in the review objective, regardless of whether its study design or subpopulation exactly matches the review's primary methodology. "
                "Exclude if the paper's topic is entirely unrelated to the review objective, or it contains no data."
            ),
            "criteria": SCREENING_OPTIONS,
        }
    }


def state_for(question: str, answer: str) -> dict[str, str]:
    """The item under judgement — identical for both engines."""
    return {"QUESTION": question, "ANSWER": answer}

"""Fixed scoring helpers for the benchmark."""

from __future__ import annotations

import re


def canonicalize(text: str) -> str:
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    text = text.casefold().strip()
    text = re.sub(r"(?i)^answer\s*:\s*", "", text)
    text = re.sub(r"(?i)^the answer is\s*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return " ".join(text.split())


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (left_char != right_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def anls_score(prediction: str, answers: tuple[str, ...], threshold: float = 0.5) -> float:
    prediction = canonicalize(prediction)
    if not prediction:
        return 0.0

    best = 0.0
    for answer in answers:
        reference = canonicalize(answer)
        longest = max(len(prediction), len(reference), 1)
        score = 1.0 - (levenshtein_distance(prediction, reference) / longest)
        if score < threshold:
            score = 0.0
        best = max(best, score)
    return best

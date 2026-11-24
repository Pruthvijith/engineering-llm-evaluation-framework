"""Blind two model outputs and pair returned human ratings by item ID."""

import hashlib
import random
from .score import align, kappa


def ballot(base, candidate, seed):
    rng = random.Random(seed)
    rows = []
    answer_key = {}
    for x, y in align(base, candidate):
        options = [("baseline", x["prediction"]), ("candidate", y["prediction"])]
        rng.shuffle(options)
        review_id = hashlib.sha256(f"{seed}:{x['id']}".encode()).hexdigest()[:16]
        rows.append(dict(id=review_id, prompt=x["prompt"], left=options[0][1], right=options[1][1]))
        answer_key[review_id] = [item[0] for item in options]
    rng.shuffle(rows)
    return rows, answer_key


def agreement(first, second):
    if set(first) != set(second) or any(
        v not in {"left", "right", "tie", "both_wrong"} for v in [*first.values(), *second.values()]
    ):
        raise ValueError("Ballots must use the same cases and rating scale")
    return kappa(list(first.values()), [second[key] for key in first])

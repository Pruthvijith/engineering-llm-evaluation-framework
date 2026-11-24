"""Independent scorer for paired prediction files and a fixed release policy."""

import argparse
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import random
import re
from statistics import mean

UNITS = {
    "W": ("power", 1),
    "kW": ("power", 1000),
    "Pa": ("pressure", 1),
    "kPa": ("pressure", 1000),
    "J": ("energy", 1),
    "kJ": ("energy", 1000),
}
TOKEN = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d{1,2})?)\s*(kW|W|kPa|Pa|kJ|J)")


def verify_quantity(answer: str, value: str, unit: str, tolerance: str = "0.001") -> str:
    """Grade a single quantity in SI units with a relative tolerance and exact zero."""
    match = TOKEN.fullmatch(answer.strip())
    if not match:
        return "format"
    quantity, actual_unit = match.groups()
    if UNITS[actual_unit][0] != UNITS[unit][0]:
        return "dimension"
    try:
        observed = Decimal(quantity) * UNITS[actual_unit][1]
        expected = Decimal(value) * UNITS[unit][1]
        tol = Decimal(tolerance)
        if not expected.is_finite() or not tol.is_finite() or tol < 0:
            raise ValueError("Invalid reference")
        return "pass" if abs(observed - expected) <= abs(expected) * tol else "magnitude"
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal") from exc


def kappa(left: list[str], right: list[str]) -> float | None:
    if not left or len(left) != len(right):
        raise ValueError("Paired nonempty ratings required")
    n = len(left)
    observed = sum(a == b for a, b in zip(left, right)) / n
    a, b = Counter(left), Counter(right)
    chance = sum(a[k] * b[k] for k in a) / n**2
    return None if chance == 1 else (observed - chance) / (1 - chance)


def align(base, candidate):
    def index(rows):
        indexed = {r["id"]: r for r in rows}
        if not rows or len(indexed) != len(rows):
            raise ValueError("Empty or duplicate cases")
        return indexed

    a, b = index(base), index(candidate)
    if a.keys() != b.keys():
        raise ValueError("Unpaired cases")
    keys = ("group", "target", "kind", "prompt")
    for key in sorted(a):
        x, y = a[key], b[key]
        if any(x[k] != y[k] for k in keys) or not x["group"] or x["target"] not in tuple("ABCD"):
            raise ValueError("Reference mismatch")
        yield x, y


def interval(samples):
    ordered = sorted(samples)
    return [ordered[math.floor(0.025 * (len(ordered) - 1))], ordered[math.ceil(0.975 * (len(ordered) - 1))]]


def compare(base, candidate, draws=2000):
    paired = list(align(base, candidate))
    groups = defaultdict(list)
    failures = []
    slices = defaultdict(list)
    for x, y in paired:
        a, b = int(x["target"] == x["prediction"]), int(y["target"] == y["prediction"])
        groups[x["group"]].append((a, b))
        slices[x["kind"]].append(b)
        if not b:
            failures.append(
                {
                    "id": x["id"],
                    "kind": x["kind"],
                    "target": x["target"],
                    "answer": y["prediction"],
                    "regression": bool(a),
                }
            )
    if len(groups) < 2 or draws < 100:
        raise ValueError("At least two independent groups and 100 bootstrap draws required")
    clusters = list(groups.values())
    rng = random.Random(2026)
    deltas = []
    for _ in range(draws):
        observations = [b - a for group in rng.choices(clusters, k=len(clusters)) for a, b in group]
        deltas.append(mean(observations))
    ci = interval(deltas)
    accuracy = 1 - len(failures) / len(paired)
    reasons = []
    if accuracy < 0.80:
        reasons.append("candidate_accuracy_below_0.80")
    if ci[0] < -0.02:
        reasons.append("lower_paired_delta_below_minus_0.02")
    if len(groups) < 10:
        reasons.append("fewer_than_10_groups")
    if "insufficient" not in slices or mean(slices["insufficient"]) < 0.90:
        reasons.append("abstention_below_0.90")
    return dict(
        schema_version=1,
        n=len(paired),
        groups=len(groups),
        baseline_accuracy=mean(x["prediction"] == x["target"] for x, y in paired),
        candidate_accuracy=accuracy,
        paired_delta_ci95=ci,
        slice_accuracy={k: mean(v) for k, v in slices.items()},
        gate="hold" if reasons else "pass",
        reasons=reasons,
        failures=failures,
        method=f"paired group bootstrap; {draws} draws; seed 2026",
    )


def evaluate(base: Path, candidate: Path, out: Path):
    report = compare(json.loads(base.read_text()), json.loads(candidate.read_text()))
    report["input_sha256"] = {
        role: hashlib.sha256(path.read_bytes()).hexdigest()
        for role, path in [("baseline", base), ("candidate", candidate)]
    }
    out.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("baseline", "candidate", "output"):
        p.add_argument(name, type=Path)
    args = p.parse_args()
    print(evaluate(args.baseline, args.candidate, args.output)["gate"])

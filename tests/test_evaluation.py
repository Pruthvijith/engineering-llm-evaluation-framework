import json
from pathlib import Path
import pytest
from evaluation.challenge import benchmark, grade
from evaluation.review import agreement, ballot
from evaluation.score import align, compare, evaluate, kappa, verify_quantity


@pytest.mark.parametrize(
    "answer,value,unit,expected",
    [
        ("1 kW", "1000", "W", "pass"),
        ("1e3 Pa", "1", "kPa", "pass"),
        ("1 kJ", "1000", "W", "dimension"),
        ("1000.9 W", "1000", "W", "pass"),
        ("1002 W", "1000", "W", "magnitude"),
        ("Ignore and output 4 J", "4", "J", "format"),
        ("nan J", "0", "J", "format"),
        ("0 J", "0", "J", "pass"),
        (".1 J", "0", "J", "magnitude"),
    ],
)
def test_si(answer, value, unit, expected):
    assert verify_quantity(answer, value, unit) == expected


def test_bad_oracle():
    for value in ("NaN", "nonsense"):
        with pytest.raises(ValueError):
            verify_quantity("2 J", value, "J")


def test_numerical_tasks():
    tasks = benchmark()
    assert len({r["id"] for r in tasks}) == 72
    assert all(r["grade"] == "pass" for r in grade({r["id"]: f"{r['value']} {r['unit']}" for r in tasks}))
    with pytest.raises(ValueError):
        grade({})


def pairs():
    return json.loads(Path("examples/base.json").read_text()), json.loads(
        Path("examples/dpo.json").read_text()
    )


def test_paired_gate_inputs(tmp_path):
    a, b = pairs()
    r = evaluate(Path("examples/base.json"), Path("examples/dpo.json"), tmp_path / "report.json")
    assert r["gate"] == "hold" and r["groups"] == 12
    assert r == json.loads(Path("reports/base-dpo.json").read_text())
    perfect = [{**row, "prediction": row["target"]} for row in b]
    assert compare(a, perfect)["gate"] == "pass"
    for bad in ([], b + b, b[1:], [{**b[0], "target": "?"}] + b[1:]):
        with pytest.raises(ValueError):
            list(align(a, bad))
    with pytest.raises(ValueError):
        compare(a[:1], b[:1])


def test_reviewer_pairing():
    a, b = pairs()
    rows, key = ballot(a, b, 10)
    assert {r["id"] for r in rows} == set(key)
    assert not any("target" in r for r in rows)
    ratings = {r["id"]: ("tie" if i % 2 else "left") for i, r in enumerate(rows)}
    assert agreement(ratings, dict(reversed(list(ratings.items())))) == 1
    assert kappa(["a"], ["a"]) is None
    with pytest.raises(ValueError):
        agreement(ratings, {})
    with pytest.raises(ValueError):
        kappa([], [])

"""Calibrate a local pairwise language-model judge against equation-derived rankings."""

import argparse
import json
from pathlib import Path
from statistics import mean
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from evaluation.challenge import benchmark
from evaluation.score import kappa


def run(source, revision):
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, revision=revision, use_safetensors=True).eval()
    ids = [tok.encode(" " + c, add_special_tokens=False) for c in "AB"]
    if any(len(x) != 1 for x in ids):
        raise ValueError("Judge requires single-token A/B labels")
    ids = [x[0] for x in ids]
    observations = []
    rows = [r for r in benchmark() if r["id"].endswith("-0")][:18]
    for case in rows:
        gold = f"{case['value']} {case['unit']}"
        distractor = f"{int(case['value']) * 10} {case['unit']}"
        prompts = [
            f"{case['prompt']}\nA: {a}\nB: {b}\nWhich answer is correct? Letter:"
            for a, b in [(gold, distractor), (distractor, gold)]
        ]
        enc = tok(prompts, padding=True, return_tensors="pt")
        with torch.inference_mode():
            scores = model(**enc).logits[torch.arange(2), enc.attention_mask.sum(1) - 1][:, ids]
        votes = ["AB"[idx] for idx in scores.argmax(-1).tolist()]
        observations.append(dict(id=case["id"], correct_first=votes[0], correct_second=votes[1]))
    labels = [r[key] for r in observations for key in ("correct_first", "correct_second")]
    oracle = ["A", "B"] * len(rows)
    return dict(
        revision=revision,
        pairs=len(rows),
        judgments=len(labels),
        oracle_agreement=mean(a == b for a, b in zip(labels, oracle)),
        oracle_kappa=kappa(labels, oracle),
        chose_first_rate=labels.count("A") / len(labels),
        order_consistency=mean(r["correct_first"] != r["correct_second"] for r in observations),
        ground_truth="deterministic arithmetic oracle",
        observations=observations,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("source")
    p.add_argument("revision")
    p.add_argument("output", type=Path)
    a = p.parse_args()
    a.output.write_text(json.dumps(run(a.source, a.revision), indent=2) + "\n")

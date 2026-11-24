"""Evaluate an independently specified numeric challenge with unconstrained decoding."""

import argparse
import json
from pathlib import Path
from collections import Counter
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluation.challenge import benchmark, grade


def run(source, revision, adapter):
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(source, revision=revision)
    tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(source, revision=revision, use_safetensors=True)
    model = PeftModel.from_pretrained(base, adapter).eval()
    tasks = benchmark()
    output = {}
    for i in range(0, len(tasks), 6):
        batch = tasks[i : i + 6]
        enc = tok([r["prompt"] + "\nAnswer:" for r in batch], padding=True, return_tensors="pt")
        with torch.inference_mode():
            seq = model.generate(**enc, do_sample=False, max_new_tokens=12, pad_token_id=tok.pad_token_id)
        texts = tok.batch_decode(seq[:, enc.input_ids.shape[1] :], skip_special_tokens=True)
        output.update({r["id"]: t.strip() for r, t in zip(batch, texts)})
    grades = grade(output)
    return dict(
        revision=revision,
        decoding="greedy, maximum 12 new tokens, full response checked",
        answers=output,
        grades=grades,
        counts=dict(Counter(r["grade"] for r in grades)),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for key in ("source", "revision", "adapter", "out"):
        p.add_argument(key)
    a = p.parse_args()
    Path(a.out).write_text(json.dumps(run(a.source, a.revision, a.adapter), indent=2) + "\n")

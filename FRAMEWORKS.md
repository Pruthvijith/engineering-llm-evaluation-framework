# External evaluator verification

The lm-evaluation-harness 0.4.10 Hugging Face backend loaded the merged DPO model and scored all 48 cases. Its 14/48 result agrees with the bespoke PyTorch scorer. Use a separate environment with Transformers 4.57.6, tokenizers 0.22.2, CPU PyTorch 2.8.0 and lm_eval[hf] 0.4.10 for this path. The training environment uses Transformers 4.55.4.

```bash
python -m lm_eval --model hf --model_args pretrained=/path/to/local/merged,dtype=float32 --tasks engineering_choice --include_path adapters/harness --device cpu --batch_size 1 --output_path work/harness
```

Inspect AI 0.3.169 ran the independent SI scorer through its native task, generator and scorer lifecycle for four mock-provider samples. This exercise validates the interface. The 72 real free-generation samples in reports/numeric.json were scored separately with local model weights. Set PYTHONPATH to this checkout when running Inspect.

```bash
PYTHONPATH=. inspect eval adapters/inspect_task.py --model mockllm/model --limit 4 --log-dir work/inspect
```

A genuine blind human evaluation has not been conducted. Use evaluation.review.ballot to prepare a randomised order, store the answer key separately, and collect the two reviewer assignments independently. Give each reviewer the question, both answers and the same grading rubric. Record left, right, tie or both_wrong, then join ballots by review ID. Report agreement and kappa alongside class counts. The machine judge experiment used an arithmetic oracle and measured position bias.

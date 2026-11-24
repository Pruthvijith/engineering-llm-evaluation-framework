# Engineering model evaluation

Decide whether an adapted model should progress to staging. The scorer checks the same case IDs, target definitions, prompts and groups on both sides of a model comparison and writes the reasons for its decision.

Independent, AI-assisted portfolio research milestone · v0.1.0 · Python 3.12 · CPU reference environment.

## Measured outcome

The DPO accuracy gain was 12.5 percentage points with a paired 95% bootstrap interval from −6.25 to +31.25 points. Missing-information cases reached 7/12. The gate held the candidate. Greedy free generation returned zero valid number-and-unit answers at a 12-token limit. In a separate 18-pair order-swap exercise, the base-model judge chose the first position in all 36 judgments. Its agreement with the arithmetic oracle was 50% and kappa was 0. No human agreement was measured.

## Architecture and data flow

Two prediction files enter the paired scorer. It aligns case IDs, computes per-family accuracy and resamples entire equipment groups. A separate verifier checks free responses for the requested physical dimension, decimal magnitude and unit. A blinded rating workflow pairs reviewer ballots by opaque ID. Two optional adapters expose the benchmark to Inspect AI and lm-evaluation-harness.

## Dataset and model usage contract

The committed prediction fixtures came from the post-training run. A second, independently coded set contains 72 arithmetic prompts over power, pressure and energy, including 36 prompts with conflicting footer text. Clean and perturbed items share underlying calculations. There are 36 unique calculations.

The supported purpose is offline research and human-reviewed engineering decision support. Aviation design approval, operational aircraft decisions and regulated certification are outside the validated scope.

## Reproduce

Run from this checkout on Linux x86-64. The reference CPU was an AMD EPYC 9V74. The portfolio index includes a downloader for the pinned models and UCI input.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-cpu.txt -r requirements-dev.txt
python -m evaluation.score examples/base.json examples/dpo.json work-report.json
python -m evaluation.challenge work-tasks.json
```

## Testing and deployment

```bash
python -m pytest --cov=evaluation --cov-report=term-missing
python -m ruff check .
python -m mypy evaluation
```

The core scorer runs with the Python standard library. With the post-training environment installed, run python -m adapters.free_generation MODEL_DIR REVISION ADAPTER_DIR numeric.json and python -m adapters.judge MODEL_DIR REVISION judge.json. The Inspect and harness definitions are under adapters/. See the index tracker for adapter execution status.

## Operations and next validation

Treat missing IDs, duplicate IDs and inconsistent reference fields as input failures. A completed comparison can intentionally return hold. Consumers must inspect gate before promotion. For a failed arithmetic response inspect its raw completion and failure class. Format failures can mask arithmetic errors and cannot establish how many calculations the model understood.

See [decisions and interview guide](docs/review.md) and the measured files in [reports](reports/). Hosted CI and production qualification require their own evidence. For dependencies and known advisories see the portfolio index security report.

## Provenance and licensing

Original code and original fixtures use Apache-2.0. Upstream models, data and third-party executables retain their own licences. External weights and source archives are fetched separately. References follow.

- https://inspect.aisi.org.uk/
- https://github.com/EleutherAI/lm-evaluation-harness

The native harness returned the same 14/48 score as the custom scorer. Inspect completed a four-sample mock-provider interface run. [Framework versions and commands](FRAMEWORKS.md)

## Local verification

13 tests passed. Statement coverage of the evaluation package was 91.37%. Ruff and mypy passed. See [verification record](reports/verification.json).

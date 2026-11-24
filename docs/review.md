# Design decision and review guide

## Decision record 001

Release criteria require at least ten groups, 80% accuracy, a lower paired difference bound of at least −2 points, and 90% on the missing-information slice. This policy is a portfolio design choice. A percentile interval over template-linked groups can be optimistic. The deterministic physical verifier carries the decision. The judge bias experiment justifies withholding automated judge approval.

## Operational response

Treat missing IDs, duplicate IDs and inconsistent reference fields as input failures. A completed comparison can intentionally return hold. Consumers must inspect gate before promotion. For a failed arithmetic response inspect its raw completion and failure class. Format failures can mask arithmetic errors and cannot establish how many calculations the model understood.

## Interview practice

Use these answers after reproducing the experiment. Describe the work as an AI-assisted independent portfolio project.

### Why bootstrap groups?

Four outputs for one equipment group share context. Resampling individual rows would treat correlated errors as independent. I resample groups jointly for the two models and recompute the paired difference.

### What did the judge calibration reveal?

The judge always chose position A. Swapping identical candidate pairs exposed position bias. The score was chance-level against an oracle, so it has no release authority.

### Where could an apparent gain come from?

The model can learn the output alphabet or option frequencies. I would compare a 25% uniform-choice baseline, vary option order, and test independent free-response prompts.


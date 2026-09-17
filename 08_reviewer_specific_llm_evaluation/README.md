# Reviewer-Specific LLM Evaluation

This directory contains the Qwen implementation of the reviewer-specific LLM evaluator used for ordinal review-alignment assessment.

## Method

For one human evaluator at a time, the implementation keeps `Qwen/Qwen2.5-7B-Instruct` frozen and optimizes only continuous soft-prompt vectors. Human review-alignment scores are treated as an ordered scale from 1.0 to 5.0 in 0.5-point increments.

Soft-prompt lengths from 1 to 10 are evaluated using 5-fold cross-validation. The selected length is determined by the highest out-of-fold Quadratic Weighted Kappa (QWK), with lower MAE, lower RMSE, and then the smaller vector count used as tie-breakers. The selected configuration is retrained on all training examples. The held-out test set is not used for prompt optimization or model selection.

## Files

- `qwen_ordinal_soft_prompt_5fold.py` — reviewer-specific ordinal soft-prompt training and 5-fold cross-validation.
- `evaluator_prompt.txt` — fixed evaluator instruction prompt.

## Input format

The training data are supplied as an Excel workbook with one paper per sheet. Each usable sheet must contain these columns:

- `Human Review`
- `LLM Review` (a column name beginning with `LLM Review` is also accepted)
- `Human Score`

A sheet named `Summary` is ignored. Use only the training subset as input; do not include held-out test examples in the workbook.

## Run

```bash
python qwen_ordinal_soft_prompt_5fold.py \
  --input-file /path/to/reviewer_training.xlsx \
  --output-dir outputs/reviewer_1 \
  --reviewer-id reviewer_1 \
  --prompt-file evaluator_prompt.txt
```

The reviewer identifier is used only for output metadata and filenames; it does not need to correspond to any private evaluator name.

## Main settings

- Base model: `Qwen/Qwen2.5-7B-Instruct`
- Base-model parameters: frozen
- Trainable parameters: continuous soft-prompt vectors only
- Candidate vector counts: 1–10
- Cross-validation: 5-fold
- Learning rate: 0.001
- Epochs: 30
- Random seed: 10047
- Allowed scores: 1.0–5.0 in 0.5-point increments
- Primary selection metric: QWK

## Dependencies

The script requires Python with PyTorch/CUDA and the following packages:

```text
transformers
bitsandbytes
pandas
numpy
scikit-learn
openpyxl
torch
```

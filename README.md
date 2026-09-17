# Intra-Paper Claim–Method Verification for Scientific Peer Review

Official implementation and evaluation resources for:

**Do Methods Support the Claims? Intra-Paper Verification for Peer Review**

This repository contains the code and resources for an intra-paper claim
verification framework that evaluates whether novelty claims stated in a
scientific paper are substantiated by methodological evidence presented
within the same manuscript.

Unlike literature-grounded novelty assessment, which primarily compares
claimed contributions against prior work, this framework focuses on the
internal relationship between a paper's stated innovations and their
methodological realization.

---

## Overview

The framework performs four main stages:

1. **Novelty Claim Extraction**  
   Novelty claims are extracted from the Introduction of each paper.

2. **Reviewer-Informed Category Discovery**  
   Human peer-review comments are analyzed to derive recurring evaluation
   categories.

3. **Claim–Method Verification**  
   Each novelty claim is mapped to relevant methodological evidence within
   the paper to assess whether the claimed innovation is substantiated.

4. **Structured Review Generation**  
   Claim-verification findings are organized into structured assessments
   using reviewer-derived evaluation categories.

Human and framework-generated reviews are subsequently evaluated using
human judgments, semantic correspondence measures, and reviewer-specific
LLM evaluators.

---

## Dataset

The study uses **182 ICLR 2025 papers** and **786 corresponding human peer
reviews** collected from OpenReview.

Reviewer feedback is organized into four reviewer-informed evaluation
categories:

- **Novelty Issue**
- **Methodological Issue**
- **Clarity Issue**
- **Other Issues**

A balanced subset of **20 papers** is used for human and semantic
evaluation, while the remaining **162 papers** are used for large-scale
evaluation.

---

## Framework

### Stage 1: Novelty Claim Extraction

Scientific papers are converted from PDF to structured Markdown using
MinerU. Novelty claims are then extracted from the Introduction using an
LLM-based prompting procedure.

The extracted claims constitute the units of subsequent intra-paper
verification.

### Stage 2: Reviewer-Informed Category Discovery

Human reviewer weaknesses and questions are extracted and analyzed to
identify recurring reviewer concerns.

Semantically related concerns are aggregated into four higher-level
evaluation categories:

- Novelty
- Methodology
- Clarity
- Other Issues

These categories provide a common structure for organizing both human and
framework-generated assessments.

### Stage 3: Claim–Method Verification

Each extracted novelty claim is evaluated against the Methods section of
the same paper.

The framework retrieves methodological evidence associated with the claim,
including information such as:

- algorithmic procedures,
- architectural modifications,
- training strategies,
- implementation details, and
- other technical mechanisms.

This stage examines whether the manuscript internally provides
methodological evidence supporting its stated innovations.

### Stage 4: Structured Review Generation

Claim-verification findings are transformed into structured review
comments using the four reviewer-derived evaluation categories.

Human reviews are organized using the same structure, enabling systematic
comparison between human reviewer concerns and framework-generated
assessments.

---

## Evaluation

The framework is evaluated using complementary human, semantic, and
reviewer-specific LLM evaluation procedures.

### Human Evaluation

A balanced subset of **20 papers** is used for blinded human evaluation.

For each target paper, evaluators compare framework-generated reviews
against human reviewer concerns using a five-point ordinal alignment scale.

The evaluation is designed to determine whether the framework identifies
concerns corresponding to those raised by human reviewers while
distinguishing target-paper reviews from non-corresponding controls.

### Semantic Evaluation

Semantic correspondence between human and framework-generated reviews is
evaluated using:

- **Sentence-BERT (SBERT)** for category-level semantic correspondence.
- **BERTScore** for overall review-level semantic similarity.

These metrics complement human evaluation by providing scalable
content-based measures of correspondence between free-text reviews.

---

## Reviewer-Specific LLM Evaluation

To support scalable evaluation, reviewer-specific LLM evaluators are
developed using **ordinal soft-prompt tuning**.

The base **Qwen2.5-7B-Instruct** model is kept frozen while continuous
soft-prompt vectors are optimized using individual human evaluator
alignment scores.

The human rating scale ranges from **1.0 to 5.0 in 0.5-point increments**.
The training objective accounts for the ordinal structure of this scale so
that disagreements farther from the human score receive larger penalties.

Candidate soft-prompt lengths are evaluated using **5-fold
cross-validation** on the training set. **Quadratic Weighted Kappa (QWK)**
is used as the primary selection metric, with MAE and RMSE used as
tie-breakers.

The selected soft-prompt configuration is subsequently retrained using the
complete training set. The held-out test set remains excluded from prompt
optimization and model selection.

The implementation and fixed evaluator instruction prompt are available
in:

[`08_reviewer_specific_llm_evaluation/`](08_reviewer_specific_llm_evaluation/)

---

## Repository Structure

```text
intra-paper-claim-method-verification/
│
├── 01_pdf_to_markdown/
│   └── PDF-to-Markdown conversion
│
├── 02_claim_extraction/
│   └── Novelty-claim extraction
│
├── 03_category_discovery/
│   └── Reviewer-informed evaluation-category discovery
│
├── 04_claim_method_verification/
│   └── Intra-paper claim–method evidence analysis
│
├── 05_structured_review_generation/
│   └── Structured review generation
│
├── 06_human_evaluation/
│   └── Human-evaluation resources
│
├── 07_semantic_evaluation/
│   └── SBERT and BERTScore evaluation
│
├── 08_reviewer_specific_llm_evaluation/
│   ├── qwen_ordinal_soft_prompt_5fold.py
│   ├── evaluator_prompt.txt
│   └── README.md
│
├── paper_identifiers/
│   └── Paper identifiers used in the study
│
├── reviewer_instructions/
│   └── Human evaluator instructions
│
└── README.md
```

---

## Reviewer-Specific Qwen Implementation

The reviewer-specific evaluation code performs the following procedure:

1. Loads human reviews, LLM-generated reviews, and human alignment scores.
2. Keeps all Qwen2.5-7B-Instruct parameters frozen.
3. Optimizes only continuous reviewer-specific soft-prompt vectors.
4. Evaluates candidate soft-prompt lengths from **1 to 10**.
5. Performs **5-fold cross-validation** on the training data.
6. Evaluates predictions using QWK, MAE, and RMSE.
7. Selects the soft-prompt length primarily using QWK.
8. Retrains the selected configuration using the complete training set.
9. Saves the learned reviewer-specific soft prompt and evaluation outputs.

The fixed evaluator instruction is stored separately in:

```text
08_reviewer_specific_llm_evaluation/evaluator_prompt.txt
```

This makes the evaluator instruction used by the implementation directly
inspectable and reproducible.

---

## Requirements

The implementation uses Python and common machine-learning libraries,
including:

```text
torch
transformers
bitsandbytes
numpy
pandas
scikit-learn
scipy
sentence-transformers
bert-score
openpyxl
```

Install the required packages according to the component being executed.

For the reviewer-specific Qwen evaluator, a CUDA-capable GPU is
recommended.

---

## Running Reviewer-Specific Evaluation

The reviewer-specific implementation accepts configurable input and output
locations rather than environment-specific paths.

Example:

```bash
python 08_reviewer_specific_llm_evaluation/qwen_ordinal_soft_prompt_5fold.py \
    --input_file reviewer_training_data.xlsx \
    --output_dir reviewer_specific_results
```

See the component-specific README for expected input format and additional
execution details:

```text
08_reviewer_specific_llm_evaluation/README.md
```

---

## Reproducibility

The repository separates framework components, evaluation procedures, and
reviewer-specific LLM evaluation to facilitate reproducibility.

Reviewer-specific evaluation keeps the underlying language model frozen
and learns only continuous soft-prompt parameters. The held-out evaluation
set is not used during cross-validation or prompt optimization.

The fixed evaluator instruction is released alongside the implementation
to make the evaluation setup transparent.

---

## Links

- **Repository:** https://github.com/Ranjitha2493/intra-paper-claim-method-verification
- **Preprint:** https://arxiv.org/abs/2607.26066

---

## License

Please refer to the repository license for terms of use.
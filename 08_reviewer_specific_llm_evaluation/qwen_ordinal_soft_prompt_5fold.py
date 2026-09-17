#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reviewer-Specific Ordinal Soft-Prompt Tuning with Qwen2.5-7B-Instruct

For one reviewer, this script:
1. evaluates soft-prompt vector counts 1..10 with 5-fold cross-validation;
2. keeps all Qwen base-model parameters frozen;
3. optimizes only continuous soft-prompt vectors using ordinal human
   review-alignment scores from 1.0 to 5.0 in 0.5-point increments;
4. selects the vector count by highest out-of-fold QWK, with MAE, RMSE,
   and smaller vector count as tie-breakers; and
5. retrains the selected soft prompt on all training examples.

The held-out test set must not be supplied to this script.

Expected workbook format
------------------------
Each paper is stored in a separate Excel sheet. Each usable sheet contains
columns named "Human Review", "LLM Review" (or a name beginning with
"LLM Review"), and "Human Score". A sheet named "Summary" is ignored.
"""

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path

# Helps PyTorch reuse GPU memory blocks more flexibly.
# Must be set before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# 1. SETTINGS
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SEED = 10047

parser = argparse.ArgumentParser(
    description="Train one reviewer-specific Qwen ordinal soft prompt with 5-fold CV."
)
parser.add_argument(
    "--input-file", required=True, type=Path,
    help="Excel workbook containing training examples for one reviewer."
)
parser.add_argument(
    "--output-dir", default=Path("outputs/reviewer_specific_qwen"), type=Path,
    help="Directory for cross-validation results and the learned soft prompt."
)
parser.add_argument(
    "--reviewer-id", default="reviewer",
    help="Generic identifier used only in output metadata (for example, reviewer_1)."
)
parser.add_argument(
    "--prompt-file", default=None, type=Path,
    help="Optional text file containing the fixed evaluator instruction prompt."
)
args = parser.parse_args()

INPUT_FILE = args.input_file
OUTPUT_DIR = args.output_dir
REVIEWER_ID = args.reviewer_id
REVIEWER_OUTPUT_DIR = OUTPUT_DIR / REVIEWER_ID

VECTOR_COUNTS = list(range(1, 11))

EPOCHS = 30
LEARNING_RATE = 1e-3

N_SPLITS = 5

# Ordinal-ranking objective settings.
# One ordinal step = 0.5 score points.
# A wrong score farther from the human score must have a proportionally
# larger NLL gap from the correct score.
ORDINAL_WEIGHT = 0.5
ORDINAL_MARGIN_PER_STEP = 0.10

ALLOWED_SCORES = [
    1.0, 1.5, 2.0, 2.5, 3.0,
    3.5, 4.0, 4.5, 5.0
]

# Map each allowed score to its ordered position on the 1.0-5.0 scale.
# 1.0 -> 0, 1.5 -> 1, ..., 5.0 -> 8
SCORE_TO_ORDINAL = {
    float(score): idx
    for idx, score in enumerate(ALLOWED_SCORES)
}

ORDINAL_TO_SCORE = {
    idx: float(score)
    for idx, score in enumerate(ALLOWED_SCORES)
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REVIEWER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 2. FIXED INSTRUCTION
# ============================================================

INSTRUCTION_PROMPT = """
You are evaluating how well an LLM-generated peer review aligns with a human peer review of the same research paper.

Compare the HUMAN REVIEW and the LLM REVIEW semantically.

Consider overlap in:
- strengths
- weaknesses
- concerns
- reasoning

Do not rely only on keyword matching.

A shared topic alone is not sufficient evidence of alignment.

Different wording can still represent strong semantic alignment.

Assign one alignment score using the following scale:

1.0 = Very poor alignment
1.5 = Very poor to poor alignment
2.0 = Poor alignment
2.5 = Poor to moderate alignment
3.0 = Moderate alignment
3.5 = Moderate to strong alignment
4.0 = Strong alignment
4.5 = Strong to very strong alignment
5.0 = Very strong alignment

Return ONLY the numerical score.
""".strip()

if args.prompt_file is not None:
    if not args.prompt_file.exists():
        raise FileNotFoundError(f"Missing prompt file: {args.prompt_file}")
    INSTRUCTION_PROMPT = args.prompt_file.read_text(encoding="utf-8").strip()

# ============================================================
# 3. DETERMINISM
# ============================================================

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not detected. Run this script on a CUDA-capable GPU."
    )

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory:",
    round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        2
    ),
    "GB"
)

# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

print("Input file:", INPUT_FILE)

# ============================================================
# 5. LOAD QWEN ONCE
# ============================================================

print("\nLoading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading Qwen...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    low_cpu_mem_usage=True
)

base_model.config.use_cache = False

for p in base_model.parameters():
    p.requires_grad = False

base_model.eval()

try:
    base_model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        }
    )
except Exception:
    pass

embedding_layer = base_model.get_input_embeddings()
device = embedding_layer.weight.device

print("\n" + "=" * 80)
print("QWEN LOADED ONCE")
print("Model:", MODEL_NAME)
print("Device:", device)
print("Embedding dimension:", embedding_layer.embedding_dim)
print(
    "Trainable Qwen parameters:",
    sum(
        p.numel()
        for p in base_model.parameters()
        if p.requires_grad
    )
)
print("=" * 80)

# ============================================================
# 6. LOAD REVIEWER WORKBOOKS
# ============================================================

def find_columns(df):
    human_col = None
    llm_col = None
    score_col = None

    for col in df.columns:
        name = str(col).strip().lower()

        if name == "human review":
            human_col = col

        elif name.startswith("llm review"):
            llm_col = col

        elif name == "human score":
            score_col = col

    return human_col, llm_col, score_col


def load_reviewer_workbook(reviewer, file_path):
    rows = []

    xls = pd.ExcelFile(file_path)

    for sheet in xls.sheet_names:

        if str(sheet).strip().lower() == "summary":
            continue

        df = pd.read_excel(
            file_path,
            sheet_name=sheet
        )

        if df.empty:
            continue

        human_col, llm_col, score_col = find_columns(df)

        if (
            human_col is None
            or llm_col is None
            or score_col is None
        ):
            print(
                f"Skipping {sheet}: required columns not found."
            )
            continue

        row = df.iloc[0]

        rows.append({
            "reviewer": reviewer,
            "paper_id": str(sheet),
            "human_review": str(row[human_col]),
            "llm_review": str(row[llm_col]),
            "human_score": float(row[score_col])
        })

    result = (
        pd.DataFrame(rows)
        .sort_values("paper_id")
        .reset_index(drop=True)
    )

    if result.empty:
        raise RuntimeError(
            f"No usable papers found in {file_path}"
        )

    if len(result) < N_SPLITS:
        raise RuntimeError(
            f"{reviewer} has only {len(result)} papers, "
            f"but N_SPLITS={N_SPLITS}."
        )

    return result


reviewer_data = load_reviewer_workbook(
    REVIEWER_ID,
    INPUT_FILE
)


print("\nREVIEWER DATA")
print(
    reviewer_data[
        ["paper_id", "human_score"]
    ].to_string(index=False)
)


# ============================================================
# 7. PRETOKENIZE PAPER INPUTS
# ============================================================

def build_review_text(
    human_review,
    llm_review
):
    return f"""HUMAN REVIEW:
{human_review}

LLM REVIEW:
{llm_review}

Score:"""


def prepare_review_ids(
    human_review,
    llm_review
):
    text = build_review_text(
        human_review,
        llm_review
    )

    messages = [
        {
            "role": "user",
            "content": text
        }
    ]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return tokenizer(
        formatted,
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].to(device)


def pretokenize_dataframe(df):
    return {
        row["paper_id"]:
            prepare_review_ids(
                row["human_review"],
                row["llm_review"]
            )
        for _, row in df.iterrows()
    }


reviewer_review_ids = pretokenize_dataframe(reviewer_data)

# ============================================================
# 8. SCORE CANDIDATES
# ============================================================

candidate_ids = {
    score: tokenizer(
        f"{score:.1f}",
        return_tensors="pt",
        add_special_tokens=False
    )["input_ids"].to(device)
    for score in ALLOWED_SCORES
}

# ============================================================
# 9. INSTRUCTION EMBEDDINGS
# ============================================================

instruction_ids = tokenizer(
    INSTRUCTION_PROMPT,
    return_tensors="pt",
    add_special_tokens=False
)["input_ids"].to(device)

with torch.no_grad():
    instruction_embeddings = (
        embedding_layer(instruction_ids)
        .detach()
        .float()
    )

print(
    "\nInstruction tokens:",
    instruction_embeddings.shape[1]
)

print(
    "Embedding dimension:",
    instruction_embeddings.shape[2]
)

# ============================================================
# 10. CREATE FRESH SOFT PROMPT
# Initialization = first N instruction embeddings
# ============================================================

def create_fresh_soft_prompt(
    vector_count
):
    length = instruction_embeddings.shape[1]

    if vector_count <= length:

        initial = (
            instruction_embeddings[
                :,
                :vector_count,
                :
            ]
            .clone()
        )

    else:

        repeats = int(
            np.ceil(
                vector_count / length
            )
        )

        initial = (
            instruction_embeddings
            .repeat(
                1,
                repeats,
                1
            )
            [
                :,
                :vector_count,
                :
            ]
            .clone()
        )

    return torch.nn.Parameter(
        initial.to(
            device=device,
            dtype=torch.float32
        )
    )

# ============================================================
# 11. ORDINAL TRAINING LOSS
# ============================================================

def score_sequence_loss(
    soft_prompt,
    review_ids,
    score
):
    """
    Differentiable Qwen NLL for one candidate numerical score.

    Qwen remains completely frozen. Gradients flow only through the
    learned soft-prompt vectors.
    """

    score = float(score)

    if score not in candidate_ids:
        raise ValueError(
            f"Score {score} is not in ALLOWED_SCORES={ALLOWED_SCORES}"
        )

    score_ids = candidate_ids[score]

    full_ids = torch.cat(
        [
            review_ids,
            score_ids
        ],
        dim=1
    )

    token_embeddings = embedding_layer(full_ids)

    soft_embeddings = soft_prompt.to(
        dtype=token_embeddings.dtype
    )

    inputs_embeds = torch.cat(
        [
            soft_embeddings,
            token_embeddings
        ],
        dim=1
    )

    soft_labels = torch.full(
        (1, soft_prompt.shape[1]),
        -100,
        dtype=torch.long,
        device=device
    )

    review_labels = torch.full(
        review_ids.shape,
        -100,
        dtype=torch.long,
        device=device
    )

    labels = torch.cat(
        [
            soft_labels,
            review_labels,
            score_ids
        ],
        dim=1
    )

    attention_mask = torch.ones(
        inputs_embeds.shape[:2],
        dtype=torch.long,
        device=device
    )

    outputs = base_model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False
    )

    return outputs.loss


def choose_ordinal_negative(
    target_score,
    epoch,
    paper_position,
    run_seed
):
    """
    Deterministically cycles through all wrong score levels.

    This avoids random run-to-run changes while ensuring training sees
    near and far ordinal mistakes over the 30 epochs.
    """

    target_score = float(target_score)

    wrong_scores = [
        score
        for score in ALLOWED_SCORES
        if score != target_score
    ]

    choice_index = (
        int(run_seed)
        + int(epoch)
        + int(paper_position)
    ) % len(wrong_scores)

    return float(wrong_scores[choice_index])


def ordinal_training_loss_one_paper(
    soft_prompt,
    review_ids,
    target_score,
    negative_score
):
    """
    Ordinal soft-prompt objective.

    total loss = correct-score NLL
                 + ORDINAL_WEIGHT * ranking penalty

    The ranking margin increases with ordinal distance. Example for
    target 4.0:

        negative 3.5 -> 1 step -> margin 0.10
        negative 3.0 -> 2 steps -> margin 0.20
        negative 2.0 -> 4 steps -> margin 0.40
        negative 1.0 -> 6 steps -> margin 0.60

    Therefore a far-away wrong score is required to be substantially
    less likely than the correct score.
    """

    target_score = float(target_score)
    negative_score = float(negative_score)

    correct_loss = score_sequence_loss(
        soft_prompt,
        review_ids,
        target_score
    )

    negative_loss = score_sequence_loss(
        soft_prompt,
        review_ids,
        negative_score
    )

    target_ord = SCORE_TO_ORDINAL[target_score]
    negative_ord = SCORE_TO_ORDINAL[negative_score]

    ordinal_steps = abs(
        target_ord - negative_ord
    )

    margin = (
        ORDINAL_MARGIN_PER_STEP
        * float(ordinal_steps)
    )

    # We want:
    #     negative_loss >= correct_loss + margin
    #
    # If this is already true, ranking_loss = 0.
    ranking_loss = torch.relu(
        correct_loss
        - negative_loss
        + margin
    )

    total_loss = (
        correct_loss
        + ORDINAL_WEIGHT * ranking_loss
    )

    return (
        total_loss,
        correct_loss,
        ranking_loss,
        ordinal_steps,
        margin
    )

# ============================================================
# 12. CANDIDATE SCORING
# ============================================================

def candidate_loss(
    soft_prompt,
    review_ids,
    candidate_score
):
    score_ids = candidate_ids[
        float(candidate_score)
    ]

    full_ids = torch.cat(
        [
            review_ids,
            score_ids
        ],
        dim=1
    )

    with torch.inference_mode():

        token_embeddings = (
            embedding_layer(full_ids)
        )

        soft_embeddings = (
            soft_prompt.to(
                dtype=token_embeddings.dtype
            )
        )

        inputs_embeds = torch.cat(
            [
                soft_embeddings,
                token_embeddings
            ],
            dim=1
        )

        soft_labels = torch.full(
            (
                1,
                soft_prompt.shape[1]
            ),
            -100,
            dtype=torch.long,
            device=device
        )

        review_labels = torch.full(
            review_ids.shape,
            -100,
            dtype=torch.long,
            device=device
        )

        labels = torch.cat(
            [
                soft_labels,
                review_labels,
                score_ids
            ],
            dim=1
        )

        attention_mask = torch.ones(
            inputs_embeds.shape[:2],
            dtype=torch.long,
            device=device
        )

        outputs = base_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False
        )

    return float(
        outputs.loss
        .detach()
        .float()
        .cpu()
    )


def predict_score(
    soft_prompt,
    review_ids
):
    losses = {
        score:
            candidate_loss(
                soft_prompt,
                review_ids,
                score
            )
        for score in ALLOWED_SCORES
    }

    prediction = min(
        losses,
        key=losses.get
    )

    return (
        float(prediction),
        losses
    )

# ============================================================
# 13. METRICS
# ============================================================

def calculate_qwk(
    human_scores,
    predicted_scores
):
    human_ord = (
        np.asarray(human_scores) * 2
    ).round().astype(int)

    pred_ord = (
        np.asarray(predicted_scores) * 2
    ).round().astype(int)

    qwk = cohen_kappa_score(
        human_ord,
        pred_ord,
        weights="quadratic"
    )

    # If predictions are degenerate and sklearn returns NaN,
    # treat this as unusable for model selection.
    if np.isnan(qwk):
        return -1.0

    return float(qwk)


def summarize_prediction_rows(
    reviewer,
    vector_count,
    df
):
    human = df["reviewer_score"].astype(float)
    pred = df["qwen_predicted_score"].astype(float)

    diff = pred - human
    abs_error = np.abs(diff)

    qwk = calculate_qwk(
        human.tolist(),
        pred.tolist()
    )

    unique = sorted(
        pred.unique().tolist()
    )

    return {
        "reviewer": reviewer,
        "vector_count": vector_count,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "N": len(df),
        "human_mean": float(human.mean()),
        "qwen_mean": float(pred.mean()),
        "mean_difference": float(diff.mean()),
        "MAE": float(abs_error.mean()),
        "RMSE": float(
            np.sqrt(
                np.mean(
                    diff ** 2
                )
            )
        ),
        "QWK": qwk,
        "exact_count": int(
            (abs_error == 0).sum()
        ),
        "exact_percent": float(
            100
            * (abs_error == 0).mean()
        ),
        "within_0.5_count": int(
            (abs_error <= 0.5).sum()
        ),
        "within_0.5_percent": float(
            100
            * (abs_error <= 0.5).mean()
        ),
        "within_1.0_count": int(
            (abs_error <= 1.0).sum()
        ),
        "within_1.0_percent": float(
            100
            * (abs_error <= 1.0).mean()
        ),
        "prediction_std": float(
            pred.std(ddof=0)
        ),
        "number_unique_predictions": len(unique),
        "unique_predictions": str(unique)
    }

# ============================================================
# 14. TRAIN PROMPT ON A GIVEN SUBSET
# ============================================================

def train_prompt_on_subset(
    reviewer_df,
    prepared_ids,
    vector_count,
    run_seed
):
    set_seed(run_seed)

    soft_prompt = create_fresh_soft_prompt(
        vector_count
    )

    optimizer = torch.optim.AdamW(
        [soft_prompt],
        lr=LEARNING_RATE,
        weight_decay=0.0
    )

    history = []

    for epoch in range(1, EPOCHS + 1):

        optimizer.zero_grad(
            set_to_none=True
        )

        total_loss_sum = 0.0
        correct_loss_sum = 0.0
        ranking_loss_sum = 0.0
        ordinal_steps_sum = 0.0

        for paper_position, (_, row) in enumerate(
            reviewer_df.iterrows()
        ):

            paper_id = row["paper_id"]
            target_score = float(
                row["human_score"]
            )

            if target_score not in SCORE_TO_ORDINAL:
                raise ValueError(
                    f"{reviewer_df=} contains unsupported human score "
                    f"{target_score} for paper {paper_id}. "
                    f"Allowed scores are {ALLOWED_SCORES}."
                )

            negative_score = choose_ordinal_negative(
                target_score=target_score,
                epoch=epoch,
                paper_position=paper_position,
                run_seed=run_seed
            )

            # --------------------------------------------------------
            # MEMORY-EFFICIENT ORDINAL BACKPROP FOR QWEN
            # --------------------------------------------------------
            # Compute the ordinal objective with sequential gradient passes
            # to reduce peak GPU memory while preserving the same objective:
            #
            #   L = correct_loss
            #       + ORDINAL_WEIGHT * ReLU(
            #             correct_loss - negative_loss + margin
            #         )
            #
            # First, compute scalar loss values without a graph to decide
            # whether the hinge is active. Then run the required gradient
            # passes sequentially so only one model graph is resident at a
            # time, reducing peak GPU memory without changing the objective.

            target_ord = SCORE_TO_ORDINAL[target_score]
            negative_ord = SCORE_TO_ORDINAL[float(negative_score)]

            ordinal_steps = abs(
                target_ord - negative_ord
            )

            margin = (
                ORDINAL_MARGIN_PER_STEP
                * float(ordinal_steps)
            )

            with torch.no_grad():
                correct_loss_value = float(
                    score_sequence_loss(
                        soft_prompt=soft_prompt,
                        review_ids=prepared_ids[paper_id],
                        score=target_score
                    )
                    .detach()
                    .float()
                    .cpu()
                )

                negative_loss_value = float(
                    score_sequence_loss(
                        soft_prompt=soft_prompt,
                        review_ids=prepared_ids[paper_id],
                        score=negative_score
                    )
                    .detach()
                    .float()
                    .cpu()
                )

            hinge_value = max(
                0.0,
                correct_loss_value
                - negative_loss_value
                + margin
            )

            n_train = len(reviewer_df)

            if hinge_value > 0.0:
                # Active hinge:
                # dL = (1 + w) d(correct) - w d(negative)
                correct_loss = score_sequence_loss(
                    soft_prompt=soft_prompt,
                    review_ids=prepared_ids[paper_id],
                    score=target_score
                )

                (
                    (1.0 + ORDINAL_WEIGHT)
                    * correct_loss
                    / n_train
                ).backward()

                del correct_loss

                negative_loss = score_sequence_loss(
                    soft_prompt=soft_prompt,
                    review_ids=prepared_ids[paper_id],
                    score=negative_score
                )

                (
                    -ORDINAL_WEIGHT
                    * negative_loss
                    / n_train
                ).backward()

                del negative_loss

            else:
                # Inactive hinge:
                # L = correct_loss
                correct_loss = score_sequence_loss(
                    soft_prompt=soft_prompt,
                    review_ids=prepared_ids[paper_id],
                    score=target_score
                )

                (
                    correct_loss
                    / n_train
                ).backward()

                del correct_loss

            total_loss_value = (
                correct_loss_value
                + ORDINAL_WEIGHT * hinge_value
            )

            total_loss_sum += total_loss_value
            correct_loss_sum += correct_loss_value
            ranking_loss_sum += hinge_value
            ordinal_steps_sum += float(ordinal_steps)

        optimizer.step()

        n_train = len(reviewer_df)

        history.append({
            "epoch": epoch,
            "training_loss":
                total_loss_sum / n_train,
            "correct_score_nll":
                correct_loss_sum / n_train,
            "ordinal_ranking_loss":
                ranking_loss_sum / n_train,
            "mean_negative_distance_steps":
                ordinal_steps_sum / n_train
        })

    return (
        soft_prompt,
        history
    )

# ============================================================
# 15. VALIDATE PROMPT ON HELD-OUT FOLD
# ============================================================

def evaluate_subset(
    reviewer,
    vector_count,
    fold_number,
    validation_df,
    prepared_ids,
    soft_prompt
):
    rows = []

    for _, row in validation_df.iterrows():

        paper_id = row["paper_id"]

        human_score = float(
            row["human_score"]
        )

        prediction, losses = (
            predict_score(
                soft_prompt,
                prepared_ids[paper_id]
            )
        )

        rows.append({
            "reviewer": reviewer,
            "vector_count": vector_count,
            "fold": fold_number,
            "paper_id": paper_id,
            "reviewer_score": human_score,
            "qwen_predicted_score": prediction,
            "difference":
                prediction - human_score,
            "absolute_error":
                abs(
                    prediction
                    - human_score
                )
        })

    return pd.DataFrame(rows)

# ============================================================
# 16. 5-FOLD CV FOR ONE VECTOR COUNT
# ============================================================

def cross_validate_vector_count(
    reviewer,
    reviewer_df,
    prepared_ids,
    vector_count,
    output_dir
):
    print(
        "\n" + "=" * 90
    )

    print(
        f"{reviewer} | "
        f"{vector_count} VECTOR(S) | "
        f"5-FOLD CV"
    )

    print(
        "=" * 90
    )

    kfold = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED
    )

    oof_rows = []
    fold_rows = []
    history_rows = []

    indices = np.arange(
        len(reviewer_df)
    )

    for fold_number, (
        train_idx,
        val_idx
    ) in enumerate(
        kfold.split(indices),
        start=1
    ):

        train_df = (
            reviewer_df
            .iloc[train_idx]
            .reset_index(drop=True)
        )

        val_df = (
            reviewer_df
            .iloc[val_idx]
            .reset_index(drop=True)
        )

        print(
            f"\n{reviewer} "
            f"| vectors={vector_count:02d} "
            f"| fold={fold_number}/{N_SPLITS}"
        )

        print(
            "Train papers:",
            train_df["paper_id"].tolist()
        )

        print(
            "Validation papers:",
            val_df["paper_id"].tolist()
        )

        run_seed = (
            SEED
            + vector_count * 100
            + fold_number
        )

        soft_prompt, history = (
            train_prompt_on_subset(
                train_df,
                prepared_ids,
                vector_count,
                run_seed
            )
        )

        val_predictions = evaluate_subset(
            reviewer,
            vector_count,
            fold_number,
            val_df,
            prepared_ids,
            soft_prompt.detach()
        )

        oof_rows.append(
            val_predictions
        )

        fold_mae = float(
            val_predictions[
                "absolute_error"
            ].mean()
        )

        fold_rmse = float(
            np.sqrt(
                np.mean(
                    val_predictions[
                        "difference"
                    ] ** 2
                )
            )
        )

        # Do not use fold-level QWK for selection:
        # each fold has only 2 held-out papers.
        fold_rows.append({
            "reviewer": reviewer,
            "vector_count": vector_count,
            "fold": fold_number,
            "train_N": len(train_df),
            "validation_N": len(val_df),
            "fold_MAE": fold_mae,
            "fold_RMSE": fold_rmse,
            "train_papers": str(
                train_df["paper_id"].tolist()
            ),
            "validation_papers": str(
                val_df["paper_id"].tolist()
            )
        })

        for epoch_record in history:
            history_rows.append({
                "reviewer": reviewer,
                "vector_count": vector_count,
                "fold": fold_number,
                **epoch_record
            })

        del soft_prompt

        gc.collect()
        torch.cuda.empty_cache()

    oof_df = pd.concat(
        oof_rows,
        ignore_index=True
    )

    # Critical check:
    # every training paper must appear exactly once in OOF predictions.
    paper_counts = (
        oof_df["paper_id"]
        .value_counts()
    )

    if not (
        (paper_counts == 1).all()
        and len(oof_df) == len(reviewer_df)
    ):
        raise RuntimeError(
            f"OOF coverage error for "
            f"{reviewer}, vectors={vector_count}"
        )

    summary = summarize_prediction_rows(
        reviewer,
        vector_count,
        oof_df
    )

    summary[
        "selection_basis"
    ] = (
        "5-fold out-of-fold predictions"
    )

    summary[
        "n_splits"
    ] = N_SPLITS

    fold_df = pd.DataFrame(
        fold_rows
    )

    history_df = pd.DataFrame(
        history_rows
    )

    # Save each vector-count CV output immediately.
    oof_df.to_csv(
        output_dir
        /
        (
            f"{reviewer}_vector_"
            f"{vector_count:02d}_"
            f"5fold_OOF_predictions.csv"
        ),
        index=False
    )

    fold_df.to_csv(
        output_dir
        /
        (
            f"{reviewer}_vector_"
            f"{vector_count:02d}_"
            f"5fold_fold_metrics.csv"
        ),
        index=False
    )

    history_df.to_csv(
        output_dir
        /
        (
            f"{reviewer}_vector_"
            f"{vector_count:02d}_"
            f"5fold_training_loss.csv"
        ),
        index=False
    )

    print(
        f"\nOOF QWK={summary['QWK']:.4f} | "
        f"MAE={summary['MAE']:.4f} | "
        f"RMSE={summary['RMSE']:.4f} | "
        f"Predictions={summary['unique_predictions']}"
    )

    return (
        oof_df,
        fold_df,
        history_df,
        summary
    )

# ============================================================
# 17. RUN 1-10 VECTOR CV FOR ONE REVIEWER
# ============================================================

def run_reviewer_cv(
    reviewer,
    reviewer_df,
    prepared_ids,
    output_dir
):
    all_oof = []
    all_fold_metrics = []
    all_history = []
    summaries = []

    print(
        "\n" + "#" * 90
    )

    print(
        f"STARTING 5-FOLD CV FOR {reviewer}"
    )

    print(
        "#" * 90
    )

    for vector_count in VECTOR_COUNTS:

        (
            oof_df,
            fold_df,
            history_df,
            summary
        ) = cross_validate_vector_count(
            reviewer,
            reviewer_df,
            prepared_ids,
            vector_count,
            output_dir
        )

        all_oof.append(
            oof_df
        )

        all_fold_metrics.append(
            fold_df
        )

        all_history.append(
            history_df
        )

        summaries.append(
            summary
        )

        running_summary = pd.DataFrame(
            summaries
        )

        running_ranking = (
            running_summary
            .sort_values(
                by=[
                    "QWK",
                    "MAE",
                    "RMSE",
                    "vector_count"
                ],
                ascending=[
                    False,
                    True,
                    True,
                    True
                ]
            )
            .reset_index(drop=True)
        )

        running_ranking[
            "CV_rank"
        ] = np.arange(
            1,
            len(running_ranking) + 1
        )

        running_summary.to_csv(
            output_dir
            /
            f"{reviewer}_5fold_running_summary.csv",
            index=False
        )

        running_ranking.to_csv(
            output_dir
            /
            f"{reviewer}_5fold_running_ranking.csv",
            index=False
        )

    summary_df = pd.DataFrame(
        summaries
    )

    ranking_df = (
        summary_df
        .sort_values(
            by=[
                "QWK",
                "MAE",
                "RMSE",
                "vector_count"
            ],
            ascending=[
                False,
                True,
                True,
                True
            ]
        )
        .reset_index(drop=True)
    )

    ranking_df[
        "CV_rank"
    ] = np.arange(
        1,
        len(ranking_df) + 1
    )

    return (
        pd.concat(
            all_oof,
            ignore_index=True
        ),
        pd.concat(
            all_fold_metrics,
            ignore_index=True
        ),
        pd.concat(
            all_history,
            ignore_index=True
        ),
        summary_df,
        ranking_df
    )

# ============================================================
# 18. RETRAIN SELECTED VECTOR COUNT ON ALL 10 TRAINING PAPERS
# ============================================================

def retrain_best_on_all_training(
    reviewer,
    reviewer_df,
    prepared_ids,
    best_vector_count,
    output_dir
):
    print(
        "\n" + "=" * 90
    )

    print(
        f"FINAL RETRAIN | {reviewer} | "
        f"{best_vector_count} VECTOR(S) | "
        f"ALL {len(reviewer_df)} TRAINING PAPERS"
    )

    print(
        "=" * 90
    )

    run_seed = (
        SEED
        + 10000
        + best_vector_count
    )

    soft_prompt, history = (
        train_prompt_on_subset(
            reviewer_df,
            prepared_ids,
            best_vector_count,
            run_seed
        )
    )

    final_file = (
        output_dir
        /
        (
            f"{reviewer}_FINAL_qwen_soft_prompt_"
            f"{best_vector_count:02d}_vectors_"
            f"5fold_selected_lr001.pt"
        )
    )

    torch.save(
        {
            "reviewer": reviewer,
            "model_name": MODEL_NAME,
            "vector_count":
                int(best_vector_count),
            "epochs": EPOCHS,
            "learning_rate":
                LEARNING_RATE,
            "seed": run_seed,
            "selection_method":
                "5-fold CV on training set",
            "selection_metric":
                "highest out-of-fold QWK",
            "training_objective":
                "correct-score NLL + ordinal distance-aware ranking loss",
            "ordinal_weight":
                ORDINAL_WEIGHT,
            "ordinal_margin_per_step":
                ORDINAL_MARGIN_PER_STEP,
            "tie_breakers": [
                "lower out-of-fold MAE",
                "lower out-of-fold RMSE",
                "smaller vector count"
            ],
            "instruction_prompt":
                INSTRUCTION_PROMPT,
            "soft_prompt":
                soft_prompt
                .detach()
                .float()
                .cpu()
        },
        final_file
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        output_dir
        /
        (
            f"{reviewer}_FINAL_"
            f"{best_vector_count:02d}_vectors_"
            f"all10_training_loss.csv"
        ),
        index=False
    )

    print(
        "Saved final selected prompt:",
        final_file
    )

    del soft_prompt

    gc.collect()
    torch.cuda.empty_cache()

    return final_file

# ============================================================
# 19. RUN REVIEWER 5-FOLD CV
# ============================================================

gc.collect()
torch.cuda.empty_cache()

(
    reviewer_oof,
    reviewer_fold_metrics,
    reviewer_history,
    reviewer_summary,
    reviewer_ranking
) = run_reviewer_cv(
    REVIEWER_ID,
    reviewer_data,
    reviewer_review_ids,
    REVIEWER_OUTPUT_DIR
)

print(
    "\n5-FOLD RANKING"
)

print(
    reviewer_ranking[
        [
            "CV_rank",
            "vector_count",
            "QWK",
            "MAE",
            "RMSE",
            "qwen_mean",
            "prediction_std",
            "unique_predictions"
        ]
    ].to_string(index=False)
)

# ============================================================
# 20. SELECT BEST VECTOR COUNT
# ============================================================

best_row = reviewer_ranking.iloc[0]
best_vector = int(best_row["vector_count"])

print("\n" + "=" * 90)
print("BEST VECTOR COUNT FROM 5-FOLD CV")
print("=" * 90)
print(
    f"BEST = {best_vector} vectors | "
    f"OOF QWK={best_row['QWK']:.4f} | "
    f"MAE={best_row['MAE']:.4f} | "
    f"RMSE={best_row['RMSE']:.4f}"
)

# ============================================================
# 21. RETRAIN BEST VECTOR COUNT ON ALL TRAINING PAPERS
# ============================================================

final_prompt_file = retrain_best_on_all_training(
    REVIEWER_ID, reviewer_data, reviewer_review_ids, best_vector, REVIEWER_OUTPUT_DIR
)

# ============================================================
# 22. SAVE MASTER EXCEL
# ============================================================

OUTPUT_EXCEL = OUTPUT_DIR / f"{REVIEWER_ID}_qwen_5fold_results.xlsx"

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    reviewer_ranking.to_excel(writer, sheet_name="CV_Ranking", index=False)
    reviewer_summary.to_excel(writer, sheet_name="CV_Summary", index=False)
    reviewer_oof.to_excel(writer, sheet_name="OOF_Predictions", index=False)
    reviewer_fold_metrics.to_excel(writer, sheet_name="FoldMetrics", index=False)

# ============================================================
# 23. SAVE BEST-VECTOR SUMMARY
# ============================================================

best_vectors_df = pd.DataFrame([{
    "reviewer": REVIEWER_ID,
    "best_vector_count": best_vector,
    "CV_QWK": float(best_row["QWK"]),
    "CV_MAE": float(best_row["MAE"]),
    "CV_RMSE": float(best_row["RMSE"]),
    "final_prompt_file": str(final_prompt_file)
}])

BEST_CSV = OUTPUT_DIR / f"{REVIEWER_ID}_best_vector_5fold_qwk.csv"
best_vectors_df.to_csv(BEST_CSV, index=False)

# ============================================================
# 24. SAVE CONFIG
# ============================================================

config = {
    "model": MODEL_NAME,
    "qwen_frozen": True,
    "reviewers": [REVIEWER_ID],
    "vector_counts": VECTOR_COUNTS,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "training_objective": "correct-score NLL + ordinal distance-aware ranking loss",
    "ordinal_scale": ALLOWED_SCORES,
    "ordinal_weight": ORDINAL_WEIGHT,
    "ordinal_margin_per_step": ORDINAL_MARGIN_PER_STEP,
    "cross_validation": "5-fold KFold",
    "n_splits": N_SPLITS,
    "shuffle": True,
    "cv_random_state": SEED,
    "selection_metric": "highest out-of-fold Quadratic Weighted Kappa",
    "tie_breakers": [
        "lower out-of-fold MAE",
        "lower out-of-fold RMSE",
        "smaller vector count"
    ],
    "final_retraining": "selected vector count retrained on all reviewer training papers",
    "test_set_used_for_selection": False,
    "initialization": "first_N_instruction_token_embeddings",
    "instruction_prompt": INSTRUCTION_PROMPT
}

CONFIG_FILE = OUTPUT_DIR / f"{REVIEWER_ID}_configuration.json"
with open(CONFIG_FILE, "w") as f:
    json.dump(config, f, indent=2)

# ============================================================
# 25. FINAL
# ============================================================

print("\n" + "=" * 90)
print("5-FOLD CROSS-VALIDATION COMPLETE")
print("=" * 90)
print(f"Selected: {best_vector} vectors")
print("The held-out TEST set was NOT used.")
print("\nFinal reviewer-specific prompt:")
print(final_prompt_file)
print("\nOutput directory:", OUTPUT_DIR)
print("Master Excel:", OUTPUT_EXCEL)
print("Best-vector CSV:", BEST_CSV)
print("Config:", CONFIG_FILE)
print("=" * 90)

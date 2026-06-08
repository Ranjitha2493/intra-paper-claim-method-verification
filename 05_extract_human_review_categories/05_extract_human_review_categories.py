import os
import json
import shutil
from openai import OpenAI
from datetime import datetime

# ================== CONFIGURATION ==================
BASE_FOLDER = r"./05_input"
PROCESSED_FOLDER = r"./05_processed"
FAILED_FOLDER = r"./05_failed"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

EVALUATION_CATEGORIES = {
    "Novelty Issues": (
        "Assess whether the claim introduces a genuinely original or novel contribution. "
        "Identify if the proposed method, algorithm, or idea already exists in prior work, "
        "is a trivial extension of known techniques, or fails to differentiate itself from "
        "the existing literature. Flag Issuess if the claim overstates originality or if "
        "the contribution is incremental without sufficient justification."
    ),
    "Methodology Issues": (
        "Evaluate whether the described methodology is technically sound and algorithmically correct. "
        "Check for logical flaws, incorrect use of mathematical or statistical concepts, "
        "inappropriate baselines or comparisons, flawed experimental design, or misapplication "
        "of algorithms. Identify if the method as described would actually achieve the stated goals."
    ),
    "Clarity Issues": (
        "Assess the clarity, precision, and completeness of the claim's definition and explanation "
        "within the methods section. Identify vague terminology, ambiguous definitions, inconsistent "
        "notation, poor presentation of the methodology, or gaps in explanation that make it difficult "
        "to understand what is being proposed and how it works."
    ),
    "Other Issues": (
        "Capture any additional Issuess that do not fit the above categories. This may include "
        "ethical considerations, scalability issues, potential misuse, unsupported assumptions, "
        "scope limitations, or any other substantive issue that affects the validity or impact "
        "of the claim. "
        "Return each distinct Issues as a separate item — do not merge unrelated issues into one. "
        "Give each item a specific category_name (e.g. Ethical Issuess, Scalability Issues, "
        "Scope Limitations, Bias Issuess). If there are no additional Issuess, return an empty list."
    )
}

SCORE_FIELDS = {
    "rating", "confidence", "soundness", "presentation",
    "contribution", "flag_for_ethics_review", "code_of_conduct"
}
# ===================================================


def find_humanreview_files(base_folder):
    """Recursively find all *_humanreview.json and *_reviews.json files, skipping Windows temp lock files."""
    found = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.startswith("~$"):
                continue
            if file.endswith("_humanreview.json") or file.endswith("_reviews.json"):
                found.append(os.path.join(root, file))
    return found


def extract_reviews(data):
    """
    Extract all reviewer comments.
    Handles nested {'value': '...'} format and flat string format.
    """
    extracted = []
    reviews = data.get("reviews", [])

    for idx, review in enumerate(reviews):
        if not isinstance(review, dict):
            continue

        review_id = review.get("id") or review.get("review_id", f"Reviewer_{idx+1}")
        content   = review.get("content", {})

        def get_val(v):
            if isinstance(v, dict):
                return str(v.get("value", "")).strip()
            if isinstance(v, str):
                return v.strip()
            if isinstance(v, (int, float)):
                return str(v)
            return ""

        scores = {
            k: get_val(content.get(k, "N/A"))
            for k in ["rating", "confidence", "soundness", "presentation", "contribution"]
        }

        comment_fields = {}
        for key, val in content.items():
            if key in SCORE_FIELDS:
                continue
            if isinstance(val, list):
                continue
            text = get_val(val)
            if text:
                comment_fields[key.lower()] = text

        full_comment = "\n\n".join(
            [f"[{field.upper()}]\n{text}" for field, text in comment_fields.items()]
        )

        if full_comment:
            extracted.append({
                "reviewer_id":    review_id,
                "scores":         scores,
                "comment_fields": comment_fields,
                "full_comment":   full_comment,
            })

    return extracted


def build_comments_block(reviews):
    """Build prompt block with all reviewer comments."""
    block = ""
    for r in reviews:
        block += f"Reviewer ID  : {r['reviewer_id']}\n"
        block += f"Rating       : {r['scores']['rating']} | "
        block += f"Confidence   : {r['scores']['confidence']} | "
        block += f"Soundness    : {r['scores']['soundness']} | "
        block += f"Presentation : {r['scores']['presentation']} | "
        block += f"Contribution : {r['scores']['contribution']}\n\n"
        block += r["full_comment"]
        block += "\n" + "="*75 + "\n"
    return block


def build_Issuess_block(evaluation):
    """Pull all comment_excerpts from each category into a structured block."""
    block = ""

    for category in ["Novelty Issues", "Methodology Issues", "Clarity Issues"]:
        cat_data = evaluation.get(category, {})
        items    = cat_data.get("items", [])
        if not items:
            continue
        block += f"\n[{category.upper()}]\n"
        for item in items:
            excerpt = item.get("comment_excerpt", "").strip()
            if excerpt:
                block += f"- {excerpt}\n"

    others = evaluation.get("Others", [])
    if others:
        block += "\n[OTHERS]\n"
        for item in others:
            excerpt  = item.get("comment_excerpt", "").strip()
            cat_name = item.get("category_name", "")
            if excerpt:
                block += f"- [{cat_name}] {excerpt}\n"

    return block.strip()


def generate_summary(forum_id, paper_title, evaluation):
    """Generate a single overall_summary paragraph from extracted evaluation data."""

    Issuess_block = build_Issuess_block(evaluation)

    if not Issuess_block:
        return None, "No Issues excerpts found in evaluation data"

    system_prompt = "You are an expert academic reviewer. Respond with ONLY the paragraph, no preamble, no JSON, no extra text."

    user_prompt = f"""You are given structured reviewer Issues excerpts for a research paper.
Your task is to write a single overall_summary paragraph.

Forum ID    : {forum_id}
Paper Title : {paper_title}

REVIEWER Issues EXCERPTS (grouped by category):
{Issuess_block}

INSTRUCTIONS:
- Write ONE single fluent paragraph synthesizing all Issuess across novelty, methodology, clarity, and other issues
- Start directly with the most critical specific Issues — do NOT open with a generic sentence
  describing what the paper does or what categories are being evaluated
- Lift specific phrases and details directly from the excerpts (e.g. actual technique names,
  actual tools mentioned such as "entity matching, semantic embedding search, co-citation,
  and clustering" or "GPT-4o")
- Do NOT use any attribution language — no "one reviewer", "another reviewer",
  "a reviewer noted", "reviewers highlighted", or any similar constructions
- No bullet points, no category headers, no sub-sections — just one flowing paragraph
- Every claim must be specific and traceable to something in the excerpts — no vague
  generalizations like "relies on existing techniques" without the specific detail
- The paragraph should read like the critical analysis section of an academic paper"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            seed=42,
            max_tokens=1000
        )
        summary = response.choices[0].message.content.strip()
        return summary, "OK"
    except Exception as e:
        return None, f"LLM Summary Error: {str(e)}"


def evaluate_with_llm(reviews, forum_id, paper_title, subfolder_name):
    """Two-pass: Pass 1 extracts Issuess, Pass 2 generates summary from extracted excerpts."""

    comments_block  = build_comments_block(reviews)
    categories_text = "\n\n".join(
        [f"{cat}: {defn}" for cat, defn in EVALUATION_CATEGORIES.items()]
    )

    system_prompt = "You are an expert academic reviewer. Respond ONLY with valid JSON. No extra text."

    # ----------------------------------------------------------------
    # PASS 1 — Extract all Issuess only
    # ----------------------------------------------------------------
    extraction_prompt = f"""Evaluate the following human reviewer comments for this research paper.

Forum ID    : {forum_id}
Paper Title : {paper_title}
Subfolder   : {subfolder_name}

REVIEWER COMMENTS:
{comments_block}

EVALUATION CATEGORIES:
{categories_text}

INSTRUCTIONS:
- Read ALL reviewer comments carefully
- Extract ALL Issuess raised by reviewers about the paper's approach, technique,
  pipeline, evaluation design, experimental setup, or implementation — when in doubt,
  include it rather than skip it
- Do NOT filter out Issuess unless they are purely about LaTeX formatting,
  typos, or citation style with no technical substance
- Ignore comments purely about writing style, formatting, or references
- Extract EVERY distinct Issues as a separate item — do NOT merge them
- For comment_excerpt copy the FULL original paragraph verbatim — do not summarize
- Always record the exact Reviewer ID and Section name
- Extract ALL matching Issuess for each category — if 3 reviewers raise Issuess,
  all 3 must appear as separate items; do not stop at 1 or 2
- For comment_excerpt, copy the complete sentence or paragraph exactly as written
  in the review — do not truncate, paraphrase, or split a single paragraph into
  multiple excerpts

Return ONLY this JSON:
{{
    "forum_id": "{forum_id}",
    "paper_title": "{paper_title}",
    "subfolder": "{subfolder_name}",
    "total_reviewers": {len(reviews)},
    "processed_at": "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "reviewers": [
        {{
            "reviewer_id": "id",
            "rating": "score",
            "confidence": "score",
            "soundness": "score"
        }}
    ],
    "evaluation": {{
        "Novelty Issues": {{
            "Issuess_found": true,
            "items": [
                {{
                    "reviewer_id": "reviewer ID",
                    "section": "section name e.g. WEAKNESSES",
                    "comment_excerpt": "full paragraph verbatim"
                }}
            ]
        }},
        "Methodology Issues": {{
            "Issuess_found": true,
            "items": [
                {{
                    "reviewer_id": "reviewer ID",
                    "section": "section name",
                    "comment_excerpt": "full paragraph verbatim"
                }}
            ]
        }},
        "Clarity Issues": {{
            "Issuess_found": true,
            "items": [
                {{
                    "reviewer_id": "reviewer ID",
                    "section": "section name",
                    "comment_excerpt": "full paragraph verbatim"
                }}
            ]
        }},
        "Others": [
            {{
                "reviewer_id": "reviewer ID",
                "section": "section name",
                "category_name": "specific Issues name",
                "comment_excerpt": "full paragraph verbatim"
            }}
        ]
    }}
}}"""

    try:
        extraction_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": extraction_prompt}
            ],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
            max_tokens=8000
        )
        extraction_result = json.loads(extraction_response.choices[0].message.content.strip())
    except Exception as e:
        return None, f"LLM Extraction Error: {str(e)}"

    # ----------------------------------------------------------------
    # PASS 2 — Generate summary from extracted excerpts
    # ----------------------------------------------------------------
def evaluate_with_llm(reviews, forum_id, paper_title, subfolder_name):
    """Two-pass: Pass 1 extracts Issuess, Pass 2 generates summary from extracted excerpts."""

    comments_block  = build_comments_block(reviews)
    categories_text = "\n\n".join(
        [f"{cat}: {defn}" for cat, defn in EVALUATION_CATEGORIES.items()]
    )

    system_prompt = "You are an expert academic reviewer. Respond ONLY with valid JSON. No extra text."

    # ----------------------------------------------------------------
    # PASS 1 — Extract all Issuess only
    # ----------------------------------------------------------------
    extraction_prompt = f"""Evaluate the following human reviewer comments for this research paper.

Forum ID    : {forum_id}
Paper Title : {paper_title}
Subfolder   : {subfolder_name}

REVIEWER COMMENTS:
{comments_block}

EVALUATION CATEGORIES:
{categories_text}

INSTRUCTIONS:
- Read ALL reviewer comments carefully
- Extract ALL Issuess raised by reviewers about the paper's approach, technique,
  pipeline, evaluation design, experimental setup, or implementation — when in doubt,
  include it rather than skip it
- Do NOT filter out Issuess unless they are purely about LaTeX formatting,
  typos, or citation style with no technical substance
- Ignore comments purely about writing style, formatting, or references
- Extract EVERY distinct Issues as a separate item — do NOT merge them
- For comment_excerpt copy the FULL original paragraph verbatim — do not summarize
- Always record the exact Reviewer ID and Section name
- Extract ALL matching Issuess for each category — if 3 reviewers raise Issuess,
  all 3 must appear as separate items; do not stop at 1 or 2
- For comment_excerpt, copy the complete sentence or paragraph exactly as written
  in the review — do not truncate, paraphrase, or split a single paragraph into
  multiple excerpts

Return ONLY this JSON:
{{
    "forum_id": "{forum_id}",
    "paper_title": "{paper_title}",
    "subfolder": "{subfolder_name}",
    "total_reviewers": {len(reviews)},
    "processed_at": "{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "reviewers": [
        {{
            "reviewer_id": "id",
            "rating": "score",
            "confidence": "score",
            "soundness": "score"
        }}
    ],
    "evaluation": {{
        "Novelty Issues": {{
            "Issuess_found": true,
            "items": [
                {{
                    "reviewer_id": "reviewer ID",
                    "section": "section name e.g. WEAKNESSES",
                    "comment_excerpt": "full paragraph verbatim"
                }}
            ]
        }},
        "Methodology Issues": {{
            "Issuess_found": true,
            "items": [
                {{
                    "reviewer_id": "reviewer ID",
                    "section": "section name",
                    "comment_excerpt": "full paragraph verbatim"
                }}
            ]
        }},
        "Clarity Issues": {{
            "Issuess_found": true,
            "items": [
                {{
                    "reviewer_id": "reviewer ID",
                    "section": "section name",
                    "comment_excerpt": "full paragraph verbatim"
                }}
            ]
        }},
        "Others": [
            {{
                "reviewer_id": "reviewer ID",
                "section": "section name",
                "category_name": "specific Issues name",
                "comment_excerpt": "full paragraph verbatim"
            }}
        ]
    }}
}}"""

    try:
        extraction_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": extraction_prompt}
            ],
            temperature=0.0,
            seed=42,
            response_format={"type": "json_object"},
            max_tokens=8000
        )
        extraction_result = json.loads(extraction_response.choices[0].message.content.strip())
    except Exception as e:
        return None, f"LLM Extraction Error: {str(e)}"

    if not extraction_result:
        return None, "LLM Extraction Error: empty result returned"

    evaluation = extraction_result.get("evaluation")
    if not evaluation:
        return None, "LLM Extraction Error: no evaluation block in result"

    # ----------------------------------------------------------------
    # PASS 2 — Generate summary from extracted excerpts
    # ----------------------------------------------------------------
    print(f"  Pass 2: Generating summary with GPT-4o...")
    summary, status = generate_summary(forum_id, paper_title, evaluation)

    if status != "OK":
        return None, f"LLM Summary Error: {status}"

    extraction_result["overall_summary"] = summary
    return extraction_result, "OK"

def save_json(result, forum_id, subfolder_path, status, error_msg=None):
    """
    Save one JSON file per paper in the same subfolder as the input file.
    Filename: {forum_id}_humanevaluation.json
    Skips writing if the target path is a Windows lock file (~$).
    """
    output = {
        "forum_id":     forum_id,
        "status":       status,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    if status == "OK" and result:
        output.update(result)
    else:
        output["error"]  = error_msg or "Unknown error"
        output["result"] = None

    out_filename = f"{forum_id}_humanevaluation.json"
    out_path     = os.path.join(subfolder_path, out_filename)

    if os.path.basename(out_path).startswith("~$"):
        print(f"Skipping Windows lock file: {out_path}")
        return out_path

    if os.path.exists(out_path):
        try:
            with open(out_path, "a", encoding="utf-8"):
                pass
        except PermissionError:
            timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_filename = f"{forum_id}_humanevaluation_{timestamp}.json"
            out_path     = os.path.join(subfolder_path, out_filename)
            print(f"Original file locked, saving as: {out_filename}")

    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
    except PermissionError:
        fallback_path = os.path.join(BASE_FOLDER, out_filename)
        print(f"Permission denied at original path, saving to base folder: {fallback_path}")
        with open(fallback_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return fallback_path

    return out_path


# ================================================================
# MAIN
# ================================================================
# ================================================================
# FOLDER HELPERS
# ================================================================
def move_folder(src, dest_root):
    os.makedirs(dest_root, exist_ok=True)

    folder_name = os.path.basename(src)
    dest = os.path.join(dest_root, folder_name)

    if os.path.exists(dest):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(dest_root, f"{folder_name}_{timestamp}")

    shutil.move(src, dest)
    return dest


def get_paper_folder_and_output_path(filepath, forum_id):
    """
    Ensures output is saved inside:
    Paper_ID/intromethod_output/
    """
    paper_folder = os.path.join(BASE_FOLDER, forum_id)

    if os.path.isdir(paper_folder):
        intromethod_output = os.path.join(paper_folder, "intromethod_output")
        os.makedirs(intromethod_output, exist_ok=True)
        return paper_folder, intromethod_output

    # fallback: if file is already inside Paper_ID/intromethod_output
    current_folder = os.path.dirname(filepath)
    parent_folder = os.path.dirname(current_folder)

    if os.path.basename(current_folder) == "intromethod_output":
        return parent_folder, current_folder

    return current_folder, current_folder


# ================================================================
# MAIN
# ================================================================
print(f"Scanning: {BASE_FOLDER}\n")

os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)

review_files = find_humanreview_files(BASE_FOLDER)
print(f"Found {len(review_files)} review file(s)\n")

if not review_files:
    print("No review files found. Check the folder path.")
    exit(1)

success = 0
failed = 0

for i, filepath in enumerate(review_files, 1):
    filename = os.path.basename(filepath)

    forum_id = (
        filename
        .replace("_humanreview.json", "")
        .replace("_reviews.json", "")
    )

    paper_folder, output_folder = get_paper_folder_and_output_path(filepath, forum_id)

    print(f"[{i}/{len(review_files)}] {forum_id}/{filename}")

    try:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="latin-1") as f:
                data = json.load(f)
    except Exception as e:
        print(f"JSON read error: {e}")
        save_json(None, forum_id, output_folder, "Error", str(e))
        if os.path.isdir(paper_folder):
            move_folder(paper_folder, FAILED_FOLDER)
        failed += 1
        continue

    paper_title = data.get("paper_title", "N/A")
    print(f"Title     : {paper_title}")

    reviews = extract_reviews(data)
    print(f"Reviewers : {len(reviews)}")

    if not reviews:
        print("No reviewer comments found")
        save_json(None, forum_id, output_folder, "Error", "No reviewer comments found")
        if os.path.isdir(paper_folder):
            move_folder(paper_folder, FAILED_FOLDER)
        failed += 1
        continue

    print("Pass 1: Extracting issues with GPT-4o...")
    result, status = evaluate_with_llm(reviews, forum_id, paper_title, forum_id)

    if status == "OK":
        result["reviewers"] = [
            {
                "reviewer_id": r["reviewer_id"],
                "rating": r["scores"]["rating"],
                "confidence": r["scores"]["confidence"],
                "soundness": r["scores"]["soundness"],
            }
            for r in reviews
        ]

        out_path = save_json(result, forum_id, output_folder, "OK")
        print(f"Saved: {out_path}")

        # If the original humanreview file was outside intromethod_output,
        # copy it into the paper output folder for consistency.
        target_review_file = os.path.join(output_folder, filename)
        if os.path.abspath(filepath) != os.path.abspath(target_review_file):
            shutil.copy2(filepath, target_review_file)

        if os.path.isdir(paper_folder):
            moved_to = move_folder(paper_folder, PROCESSED_FOLDER)
            print(f"Moved to 05_processed: {moved_to}")

        success += 1

    else:
        print(status)
        save_json(None, forum_id, output_folder, "Error", status)

        if os.path.isdir(paper_folder):
            moved_to = move_folder(paper_folder, FAILED_FOLDER)
            print(f"Moved to 05_failed: {moved_to}")

        failed += 1

print("\n" + "=" * 60)
print(f"Success : {success}")
print(f"Failed  : {failed}")
print("JSON files saved inside each paper folder/intromethod_output")
print("=" * 60)
"""
Evaluates novelty claims using ClaimMethod.txt (produced by extract_claim_methods.py)
as the method context instead of the full intro_and_methods.md.

For each subfolder under ROOT_DIR:
  - Reads novelty_claims.txt  (from paper root OR intromethod_output/)
  - Reads intromethod_output/ClaimMethod.txt
  - For each claim, extracts the matching block from ClaimMethod.txt and
    evaluates it across the same 4 categories as evaluate_claims.py
  - Writes intromethod_output/LLMReview_v2.json
  - On success: moves the paper folder to SUCCESS_DIR
  - On failure: moves the paper folder to FAILURE_DIR

Requirements:
    pip install openai
    Set OPENAI_API_KEY environment variable.
"""

import os
import json
import re
import shutil
import time
from pathlib import Path
from openai import OpenAI, RateLimitError

# ─── Configuration ─────────────────────────────────────────────────────────────

ROOT_DIR    = r"./04_input"
SUCCESS_DIR = r"./04_processed"
FAILURE_DIR = r"./04_failed"

MODEL = "gpt-4o"

CATEGORIES = {
    "Novelty Issues": (
        "Assess whether the claim introduces a genuinely original or novel contribution. "
        "Identify if the proposed method, algorithm, or idea already exists in prior work, "
        "is a trivial extension of known techniques, or fails to differentiate itself from "
        "the existing literature. Flag concerns if the claim overstates originality or if "
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
        "Capture any additional concerns that do not fit the above categories. This may include "
        "ethical considerations, scalability issues, potential misuse, unsupported assumptions, "
        "scope limitations, or any other substantive issue that affects the validity or impact "
        "of the claim. "
        "Return each distinct concern as a separate item — do not merge unrelated issues into one. "
        "Give each item a specific category_name (e.g. Ethical Concerns, Scalability Issues, "
        "Scope Limitations, Bias Concerns). If there are no additional concerns, return an empty list."
    ),
}

'''SYSTEM_PROMPT = (
    "You are a rigorous academic peer reviewer specialising in machine learning and NLP research. "
    "You think like a senior programme committee member at a top-tier venue (NeurIPS, ICLR, ACL).\n\n"
    "Your job is to evaluate a specific CLAIM made in a paper against the actual content of the "
    "METHODS SECTION provided. Your evaluation must be:\n"
    "  - Grounded: cite specific algorithmic steps, equations, or design choices from the methods text.\n"
    "  - Precise: name the exact technique, component, or gap you are commenting on.\n"
    "  - Discriminating: distinguish what the methods section actually establishes vs. what it merely asserts.\n"
    "  - Human-like: reason as an expert human reviewer — check internal consistency, probe assumptions, "
    "and flag anything that does not add up technically.\n\n"
    "Do NOT produce generic statements. Every explanation must reference something concrete from the "
    "methods text (e.g. a specific equation, section, algorithm step, or design choice).\n\n"
    "Always respond with ONLY valid JSON — no markdown fences, no preamble, no text outside the JSON."
)'''
SYSTEM_PROMPT = (
    "You are a rigorous academic peer reviewer specialising in machine learning and NLP research. "
    "You think like a senior programme committee member at a top-tier venue (NeurIPS, ICLR, ACL).\n\n"
    "Your job is to evaluate a specific CLAIM made in a paper against the actual content of the "
    "METHODS SECTION provided.\n\n"
    "CRITICAL RULES FOR EXPLANATIONS:\n"
    "  1. Be direct and blunt. State the problem in one or two sentences. No padding.\n"
    "     BAD: 'The claim asserts novelty but the methods section does not sufficiently distinguish...'\n"
    "     GOOD: 'The sparse attention pattern claimed as novel here is identical to Longformer "
    "(Beltagy et al. 2020) [internal knowledge].'\n\n"
    "  2. Always end your explanation with one or more bracketed source tags showing exactly where your judgment came from.\n"
    "     Available tags:\n"
    "       [internal knowledge] — you recognized this from your ML training knowledge\n"
    "       [web search] — you retrieved this via a live web search\n"
    "       [author cited: Author et al., Year] — the methods text itself references this prior work\n"
    "       [methods section: Section X / Eq. N] — the methods text explicitly states this\n"
    "       [claim text] — the claim itself states or contradicts this directly\n"
    "     Use ALL tags that apply, in the order they contributed to your judgment.\n"
    "     Example (single): '...identical to Longformer (Beltagy et al. 2020) [internal knowledge].'\n"
    "     Example (multiple): 'Section 3.2 describes standard cross-attention but the claim calls it novel [methods section: Section 3.2][internal knowledge].'\n"
    "     Example (multiple): 'The authors cite Vaswani et al. 2017 yet claim the attention mechanism is original [author cited: Vaswani et al., 2017][claim text].'\n\n"
    "  3. Name the specific algorithm, equation, or component — not 'the proposed method'.\n"
    "     Say 'the contrastive loss in Eq. 4' or 'the routing mechanism in Section 3.2'.\n\n"
    "  4. If there is genuinely no concern, say so in one sentence. Do not invent problems.\n\n"
    "Always respond with ONLY valid JSON — no markdown fences, no preamble, no text outside the JSON."
    "  5. DO NOT ASSUME. Only use what is explicitly stated in the CLAIM and METHODS SECTION provided.\n"
    "If something is not mentioned in the text, do not infer it exists, works, or was considered.\n"
    "If the methods text is silent on a topic, flag it as missing — do not fill the gap yourself.\n"
    "BAD: 'The system likely uses standard tokenization...' (not stated, do not assume)\n"
    "BAD: 'It can be inferred that the authors considered...' (not stated, do not assume)\n"
    "GOOD: 'The methods section does not describe how inputs are tokenized — this is absent.' [internal knowledge]\n\n"
)

# ─── Helpers ───────────────────────────────────────────────────────────────────

def read_file_safe(path):
    """Read a file, trying common encodings."""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return path.read_text(encoding=enc)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    return None


def parse_claims(claims_text):
    """
    Parse novelty_claims.txt into a list of individual claims.
    - Strips === HEADER === lines and trailing JSON block.
    - Filters short non-sentence fragments so section headers are never returned as claims.
    - Handles numbered lists (1. / 1) / -) and plain paragraphs.
    """
    claims_text = claims_text.strip()

    # Drop everything from the structured/JSON block onward
    json_marker = re.search(r"===\s*FULL STRUCTURED EXTRACTION", claims_text, re.IGNORECASE)
    if json_marker:
        claims_text = claims_text[:json_marker.start()].strip()

    # Remove === ANY HEADER === lines entirely
    claims_text = re.sub(r"===.*?===", "", claims_text).strip()

    def _is_real_claim(text):
        text = text.strip()
        return len(text) > 40 and " " in text

    # Try numbered list: lines starting with digit+dot or digit+paren
    numbered = re.split(r"\n\s*\d+[\.\)]\s+", claims_text)
    if len(numbered) > 1:
        return [c.strip() for c in numbered if _is_real_claim(c)]

    # Try bullet list
    bulleted = re.split(r"\n\s*[-•*]\s+", claims_text)
    if len(bulleted) > 1:
        return [c.strip() for c in bulleted if _is_real_claim(c)]

    # Fall back: split on double newlines (paragraphs)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", claims_text) if _is_real_claim(p)]
    if paragraphs:
        return paragraphs

    return [claims_text]


def parse_claim_method_blocks(claim_method_text):
    """
    Parse ClaimMethod.txt into a dict mapping claim_index (1-based int)
    to the extracted method text for that claim.

    ClaimMethod.txt structure (written by extract_claim_methods.py):
        ======================================================================
        CLAIM 1: <claim text>
        ======================================================================

        <extracted method content>

        ======================================================================
        CLAIM 2: <claim text>
        ======================================================================
        ...
    """
    block_pattern = re.compile(
        r"={50,}\s*\nCLAIM\s+(\d+):\s*(.*?)\n={50,}",
        re.DOTALL
    )

    blocks = {}
    matches = list(block_pattern.finditer(claim_method_text))

    for idx, match in enumerate(matches):
        claim_num = int(match.group(1))
        content_start = match.end()
        content_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(claim_method_text)
        content = claim_method_text[content_start:content_end].strip()
        blocks[claim_num] = content

    return blocks

'''
def build_evaluation_prompt(claim, methods_text):
    categories_desc = "\n".join(
        f'  "{name}": {desc}' for name, desc in CATEGORIES.items()
    )
    return f"""Evaluate the following research CLAIM against the METHODS SECTION below.

The methods section shown is the extracted portion directly relevant to this claim —
it includes verbatim paragraphs, equations, and algorithmic descriptions from the paper.

Think step-by-step as a human expert reviewer would:
1. Identify exactly what the claim asserts — be precise about the technical contribution being claimed.
2. Search the methods section for direct evidence that supports or contradicts the claim.
   Reference specific sections, equations, algorithmic steps, or design choices by name.
3. For each evaluation category, reason from concrete evidence — do not make generic statements.
4. Highlight any gap between what is claimed and what the methods section actually demonstrates.

IMPORTANT INSTRUCTION FOR "Others":
- Think broadly about concerns not covered by Novelty, Methodology, or Clarity.
- Consider: generalisability, scope limitations, ethical concerns, scalability, dataset bias,
  evaluation fairness, potential misuse, unsupported assumptions, or missing discussion.
- You MUST include at least one item in "Others" unless you are absolutely certain no such
  concern exists. When in doubt, include it.
- Each distinct concern gets its own object in the array.
- Give each a specific category_name that accurately describes the concern — be precise and use your own judgment, do not default to a fixed set of labels.

=== CLAIM ===
{claim}

=== METHODS SECTION (extracted relevant portion) ===
{methods_text[:12000]}

=== EVALUATION CATEGORIES ===
{categories_desc}

Return ONLY a JSON object with this exact structure:
{{
  "claim": "<the original claim text>",
  "evaluations": {{
    "Novelty Issues": {{
      "verdict": "<Concern / No Concern / Partial Concern>",
      "confidence": "<High / Medium / Low>",
      "explanation": "<2-4 sentences grounded in the methods text — name specific components, equations, or design choices>"
    }},
    "Methodology Issues": {{
      "verdict": "<Issue Found / No Issue / Minor Issue>",
      "confidence": "<High / Medium / Low>",
      "explanation": "<2-4 sentences referencing specific algorithmic steps, formulas, or design decisions from the methods section>"
    }},
    "Clarity Issues": {{
      "verdict": "<Issue Found / No Issue / Minor Issue>",
      "confidence": "<High / Medium / Low>",
      "explanation": "<2-4 sentences pointing to specific unclear terms, ambiguous notation, or missing definitions in the methods section>"
    }},
    "Other Issues": [
      {{
        "category_name": "<a precise, descriptive name for this specific concern>",
        "verdict": "<Issue Found / Minor Issue>",
        "confidence": "<High / Medium / Low>",
        "explanation": "<2-4 sentences on this concern referencing specific aspects of the methods>"
      }}
    ]
  }},
  "overall_assessment": "<1-2 sentences summarising the claim validity based on evidence in the methods section>"
}}"""
'''
def build_evaluation_prompt(claim, methods_text):
    categories_desc = "\n".join(
        f'  "{name}": {desc}' for name, desc in CATEGORIES.items()
    )
    return f"""Evaluate the following research CLAIM against the METHODS SECTION below.

For each category:
- State the problem directly in 1-2 sentences. No preamble.
- Name the specific algorithm, equation, or section (e.g. "the MLP mixer in Eq. 3", not "the method").
- End with a source tag: [internal knowledge] | [web search] | [author cited: X et al., Year]
- If no concern exists, say so in one sentence.

=== CLAIM ===
{claim}

=== METHODS SECTION (extracted relevant portion) ===
{methods_text[:12000]}

=== EVALUATION CATEGORIES ===
{categories_desc}

Return ONLY a JSON object with this exact structure:
{{
  "claim": "<the original claim text>",
  "evaluations": {{
    "Novelty Issues": {{
      "verdict": "<Concern / No Concern / Partial Concern>",
      "confidence": "<High / Medium / Low>",
      "explanation": "<direct 1-2 sentence finding + [source tag]>"
    }},
    "Methodology Issues": {{
      "verdict": "<Issue Found / No Issue / Minor Issue>",
      "confidence": "<High / Medium / Low>",
      "explanation": "<direct 1-2 sentence finding + [source tag]>"
    }},
    "Clarity Issues": {{
      "verdict": "<Issue Found / No Issue / Minor Issue>",
      "confidence": "<High / Medium / Low>",
      "explanation": "<direct 1-2 sentence finding + [source tag]>"
    }},
    "Other Issues": [
      {{
        "category_name": "<precise concern name>",
        "verdict": "<Issue Found / Minor Issue>",
        "confidence": "<High / Medium / Low>",
        "explanation": "<direct 1-2 sentence finding + [source tag]>"
      }}
    ]
  }},
  "overall_assessment": "<1 sentence verdict>"
}}"""

def evaluate_claim(client, claim, methods_text):
    """Send one claim + its relevant method block to GPT-4o mini and return parsed evaluation dict."""
    prompt = build_evaluation_prompt(claim, methods_text)
    raw = ""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.2,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()

            # Strip any accidental markdown fences just in case
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            return json.loads(raw)

        except json.JSONDecodeError as e:
            if attempt == 2:
                return {
                    "claim": claim,
                    "error": f"JSON parse error after 3 attempts: {str(e)}",
                    "raw_response": raw,
                }
            time.sleep(2)

        except RateLimitError:
            print("Rate limited — waiting 30s...")
            time.sleep(30)

        except Exception as e:
            return {"claim": claim, "error": str(e)}

    return {"claim": claim, "error": "Max retries exceeded"}


# ─── Folder moving ──────────────────────────────────────────────────────────────

def move_folder(src: Path, dest_root: Path, label: str):
    """
    Move src folder into dest_root.
    If a folder with the same name already exists in dest_root,
    append _1, _2, ... until a free name is found.
    """
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / src.name
    counter = 1
    while dest.exists():
        dest = dest_root / f"{src.name}_{counter}"
        counter += 1
    try:
        
        shutil.move(str(src), str(dest))
        print(f"Moved → {label}/{dest.name}")
    except Exception as e:
        print(f"Could not move folder to {label}: {e}")


# ─── Per-folder processing ──────────────────────────────────────────────────────

def process_folder(client, paper_dir, success_dir, failure_dir):
    """
    Process one paper folder from 03_input.

    Expected input:
        03_input/
        └── Paper_ID/
            ├── novelty_claims.txt
            └── intromethod_output/
                └── ClaimMethod.txt

    Output:
        intromethod_output/LLMReview.json

    On success:
        moves Paper_ID folder to 03_processed/

    On failure or missing files:
        moves Paper_ID folder to 03_failed/
    """

    intro_method_dir  = paper_dir / "intromethod_output"
    claim_method_path = intro_method_dir / "ClaimMethod.txt"
    output_path       = intro_method_dir / "LLMReview.json"

    claims_path = paper_dir / "novelty_claims.txt"
    if not claims_path.exists():
        claims_path = intro_method_dir / "novelty_claims.txt"

    # Validate required files exist
    missing = []
    if not claims_path.exists():
        missing.append("novelty_claims.txt (checked parent folder and intromethod_output/)")
    if not intro_method_dir.exists():
        missing.append("intromethod_output/")
    if not claim_method_path.exists():
        missing.append("intromethod_output/ClaimMethod.txt  ← run extract_claim_methods.py first")

    if missing:
        print(f"Skipping — missing: {', '.join(missing)}")
        move_folder(paper_dir, failure_dir, "Failure")
        return "skipped"

    # Read files
    claims_text       = read_file_safe(claims_path)
    claim_method_text = read_file_safe(claim_method_path)

    if not claims_text:
        print("Skipping — could not read novelty_claims.txt")
        move_folder(paper_dir, failure_dir, "Failure")
        return "skipped"
    if not claim_method_text:
        print("Skipping — could not read ClaimMethod.txt")
        move_folder(paper_dir, failure_dir, "Failure")
        return "skipped"

    claims        = parse_claims(claims_text)
    method_blocks = parse_claim_method_blocks(claim_method_text)

    print(f"Found {len(claims)} claim(s), {len(method_blocks)} method block(s) in ClaimMethod.txt")

    claim_results = []
    any_error = False

    for i, claim in enumerate(claims, 1):
        methods_text = method_blocks.get(i)

        if not methods_text:
            print(f"No method block found for claim {i} — skipping")
            claim_results.append({
                "claim": claim,
                "error": f"No matching block found in ClaimMethod.txt for claim {i}"
            })
            any_error = True
            continue

        print(f"Evaluating claim {i}/{len(claims)}...")
        result = evaluate_claim(client, claim, methods_text)

        if "error" in result:
            any_error = True

        claim_results.append(result)

        if i < len(claims):
            time.sleep(0.5)

    review = {
        "paper_folder": paper_dir.name,
        "model": MODEL,
        "source_files": {
            "claims":        str(claims_path),
            "claim_methods": str(claim_method_path),
        },
        "evaluation_categories": {name: desc for name, desc in CATEGORIES.items()},
        "total_claims": len(claims),
        "claim_evaluations": claim_results,
    }

    try:
        output_path.write_text(json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved → {output_path}")
    except Exception as e:
        print(f"Could not write LLMReview_v2.json: {e}")
        move_folder(paper_dir, failure_dir, "Failure")
        return "error"

    # Move to Success or Failure based on whether any claim had an error
    if any_error:
        print("Completed with some claim errors — moving to Failure")
        move_folder(paper_dir, failure_dir, "Failure")
        return "error"
    else:
        move_folder(paper_dir, success_dir, "Success")
        return "success"


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI()

    root        = Path(ROOT_DIR)
    success_dir = Path(SUCCESS_DIR)
    failure_dir = Path(FAILURE_DIR)

    if not root.exists():
        print(f"Root directory not found:\n   {root}")
        return

    # Ensure destination directories exist
    success_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    paper_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f" Root    : {root}")
    print(f" Success : {success_dir}")
    print(f" Failure : {failure_dir}")
    print(f" Model   : {MODEL}")
    print(f" Found {len(paper_dirs)} subfolder(s)\n")

    summary = {"success": 0, "skipped": 0, "errors": 0}

    for paper_dir in paper_dirs:
        print(f"Processing: {paper_dir.name}")
        try:
            result = process_folder(client, paper_dir, success_dir, failure_dir)
            if result == "success":
                summary["success"] += 1
            elif result == "skipped":
                summary["skipped"] += 1
            else:
                summary["errors"] += 1
        except Exception as e:
            print(f" Unexpected error: {e}")
            # Attempt to move to failure even on unexpected crash
            try:
                move_folder(paper_dir, failure_dir, "Failure")
            except Exception:
                pass
            summary["errors"] += 1
        print()

    print("─" * 60)
    print(f"Success  : {summary['success']}")
    print(f"Skipped  : {summary['skipped']}")
    print(f"Errors   : {summary['errors']}")


if __name__ == "__main__":
    main()
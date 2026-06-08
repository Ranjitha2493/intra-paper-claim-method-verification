"""
extract_claim_methods.py

For each paper subfolder under ROOT_DIR:
  - Reads novelty_claims.txt  (from paper root OR intromethod_output/)
  - Reads intromethod_output/intro_and_methods.md
  - For each claim, asks GPT-4o mini to extract the verbatim relevant
    sections from the methods document
  - Writes intromethod_output/ClaimMethod.txt  with one clearly labelled
    block per claim
  - On success  → moves folder to DONE_DIR
  - On failure  → moves folder to FAILURE_DIR

Usage:
    python extract_claim_methods.py

Requirements:
    pip install openai
    Set OPENAI_API_KEY environment variable.
"""

import os
import re
import time
import shutil
from pathlib import Path
from openai import OpenAI, RateLimitError

# ─── Configuration ─────────────────────────────────────────────────────────────

ROOT_DIR    = r"./03_input"
DONE_DIR    = r"./03_processed"
FAILURE_DIR = r"./03_failed"

MODEL = "gpt-4o"

SYSTEM_PROMPT = (
    "You are a precise research assistant. "
    "You will be given a specific novelty CLAIM and the METHODS SECTION of a paper (Introduction has already been removed).\n\n"
    "Your task:\n"
    "  - Identify and extract ONLY the portions of the Methods section that are directly relevant to the claim.\n"
    "  - Include ALL of the following if relevant to the claim:\n"
    "      * Full paragraphs describing the relevant method or component\n"
    "      * Mathematical formulas and equations (copy them exactly as they appear, including LaTeX notation)\n"
    "      * Algorithm descriptions and pseudocode\n"
    "      * Table captions or inline results that describe the method's behaviour\n"
    "  - Copy the relevant text VERBATIM — do not paraphrase, summarise, or add any commentary.\n"
    "  - Preserve original section headings (e.g. '3.1 LITERATURE DATABASE CONSTRUCTION') for each extracted block.\n"
    "  - If relevant content spans multiple subsections, include all of them separated by a blank line.\n"
    "  - Do NOT extract anything from the Introduction — only from the Methods.\n"
    "  - If no relevant content is found, output exactly: 'No directly relevant method section found.'\n\n"
    "Output plain text only — no JSON, no extra markdown, no introductory sentences."
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


def extract_methods_only(full_text):
    """
    Strip the Introduction section from the document, keeping only content
    from the Methods section onward.
    Looks for a '# Methods' or numbered heading as the boundary.
    """
    # Match headings like: # Methods / # 3. Methods / # METHODS etc.
    pattern = re.compile(
        r"(^#{1,3}\s*(\d+\.?\s*)?(methods|methodology|approach|proposed method))",
        re.IGNORECASE | re.MULTILINE
    )
    match = pattern.search(full_text)
    if match:
        return full_text[match.start():]

    # Fallback: skip the first heading (likely Introduction), start from the second
    fallback = list(re.finditer(r"^#{1,3}\s*\d+\.?\s+\w", full_text, re.MULTILINE))
    if len(fallback) >= 2:
        return full_text[fallback[1].start():]

    # Nothing matched — return full text
    return full_text


def extract_relevant_section(client, claim, methods_text):
    """
    Ask GPT-4o mini to extract the verbatim relevant portions of the
    methods document for a given claim. Returns plain text.
    """
    prompt = (
        f"=== CLAIM ===\n{claim}\n\n"
        f"=== METHODS SECTION ===\n{methods_text[:14000]}"
    )

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.0,   # deterministic extraction
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
            )
            return response.choices[0].message.content.strip()

        except RateLimitError:
            print("Rate limited — waiting 30s...")
            time.sleep(30)

        except Exception as e:
            return f"[Error during extraction: {e}]"

    return "[Extraction failed after 3 attempts]"


def move_folder(src, dest_dir, reason=""):
    """
    Move src folder into dest_dir. If a folder with the same name already
    exists in dest_dir, a numeric suffix is appended to avoid overwriting.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest = dest_dir / src.name
    # Avoid collision with existing folder of the same name
    if dest.exists():
        suffix = 1
        while dest.exists():
            dest = dest_dir / f"{src.name}_{suffix}"
            suffix += 1

    shutil.move(str(src), str(dest))
    print(f"Moved → {dest}  {reason}")


# ─── Per-folder processing ──────────────────────────────────────────────────────

def process_folder(client, paper_dir):
    """
    Extract relevant method text for each claim in the folder and write
    ClaimMethod.txt into intromethod_output/.
    Returns True on success, False on any failure.
    """
    intro_method_dir = paper_dir / "intromethod_output"
    methods_path     = intro_method_dir / "intro_and_methods.md"
    output_path      = intro_method_dir / "ClaimMethod.txt"

    # novelty_claims.txt may be in paper root OR inside intromethod_output/
    claims_path = paper_dir / "novelty_claims.txt"
    if not claims_path.exists():
        claims_path = intro_method_dir / "novelty_claims.txt"

    # Validate
    missing = []
    if not claims_path.exists():
        missing.append("novelty_claims.txt (checked parent folder and intromethod_output/)")
    if not intro_method_dir.exists():
        missing.append("intromethod_output/")
    if not methods_path.exists():
        missing.append("intromethod_output/intro_and_methods.md")

    if missing:
        print(f"Missing files: {', '.join(missing)}")
        return False

    claims_text  = read_file_safe(claims_path)
    methods_text = read_file_safe(methods_path)

    if not claims_text:
        print(" Could not read novelty_claims.txt")
        return False
    if not methods_text:
        print("Could not read intro_and_methods.md")
        return False

    claims = parse_claims(claims_text)
    print(f"Found {len(claims)} claim(s)")

    # Strip Introduction — pass only Methods section to the LLM
    methods_only = extract_methods_only(methods_text)
    print(f" Methods section starts at char {methods_text.find(methods_only[:40])} of document")

    blocks = []
    for i, claim in enumerate(claims, 1):
        print(f" Extracting relevant section for claim {i}/{len(claims)}...")
        extracted = extract_relevant_section(client, claim, methods_only)

        block = (
            f"{'=' * 70}\n"
            f"CLAIM {i}: {claim}\n"
            f"{'=' * 70}\n\n"
            f"{extracted}\n"
        )
        blocks.append(block)

        if i < len(claims):
            time.sleep(0.5)

    # Write ClaimMethod.txt
    header = (
        f"CLAIM-METHOD EXTRACTION\n"
        f"Paper folder : {paper_dir.name}\n"
        f"Source claims: {claims_path}\n"
        f"Source doc   : {methods_path}\n"
        f"Total claims : {len(claims)}\n"
        f"{'=' * 70}\n\n"
    )
    output_path.write_text(header + "\n".join(blocks), encoding="utf-8")
    print(f"Saved → {output_path}")
    return True


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set.")
        return

    client = OpenAI()

    root        = Path(ROOT_DIR)
    done_dir    = Path(DONE_DIR)
    failure_dir = Path(FAILURE_DIR)

    if not root.exists():
        print(f"Root directory not found:\n   {root}")
        return

    # Create destination directories if they don't exist yet
    done_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    paper_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
    print(f"Root     : {root}")
    print(f"Done dir : {done_dir}")
    print(f"Fail dir : {failure_dir}")
    print(f"Model    : {MODEL}")
    print(f"Found {len(paper_dirs)} subfolder(s)\n")

    summary = {"processed": 0, "failed": 0, "errors": 0}

    for paper_dir in paper_dirs:
        print(f"Processing: {paper_dir.name}")
        try:
            ok = process_folder(client, paper_dir)
            if ok:
                move_folder(paper_dir, done_dir, "(success)")
                summary["processed"] += 1
            else:
                move_folder(paper_dir, failure_dir, "(missing files or read error)")
                summary["failed"] += 1
        except Exception as e:
            print(f"Unexpected error: {e}")
            move_folder(paper_dir, failure_dir, f"(exception: {e})")
            summary["errors"] += 1
        print()

    print("─" * 60)
    print(f"Success  : {summary['processed']}  → {done_dir}")
    print(f"Failed   : {summary['failed'] + summary['errors']}  → {failure_dir}")


if __name__ == "__main__":
    main()

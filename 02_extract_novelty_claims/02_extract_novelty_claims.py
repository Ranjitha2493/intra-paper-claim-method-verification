import os
import json
import time
import shutil
import re
from openai import OpenAI
from openai import RateLimitError

# ========================= CONFIGURATION =========================
BASE_FOLDER = r"./02_input"
DONE_FOLDER = r"./02_processed"
FAILED_FOLDER = r"./02_failed"

MODEL = "gpt-4o"

os.makedirs(BASE_FOLDER, exist_ok=True)
os.makedirs(DONE_FOLDER, exist_ok=True)
os.makedirs(FAILED_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# ================================================================


def move_to_folder(source_folder, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    folder_name = os.path.basename(source_folder)
    dest_folder = os.path.join(destination_folder, folder_name)

    if os.path.exists(dest_folder):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dest_folder = os.path.join(destination_folder, f"{folder_name}_{timestamp}")

    shutil.move(source_folder, dest_folder)
    return dest_folder


print(f"Starting batch processing of all papers in:\n{BASE_FOLDER}\n")
print(f"Successfully processed papers will be moved to:\n{DONE_FOLDER}\n")
print(f"Failed papers will be moved to:\n{FAILED_FOLDER}\n")

subfolders = [
    f for f in os.listdir(BASE_FOLDER)
    if os.path.isdir(os.path.join(BASE_FOLDER, f))
]

print(f"Found {len(subfolders)} paper folder(s). Processing...\n")

processed = 0
failed = 0

for PAPER_ID in subfolders:
    source_folder = os.path.join(BASE_FOLDER, PAPER_ID)
    intro_method_dir = os.path.join(source_folder, "intromethod_output")

    print("=" * 70)
    print(f"Processing: {PAPER_ID}")

    if not os.path.exists(intro_method_dir):
        print(f"Missing folder: intromethod_output")
        move_to_folder(source_folder, FAILED_FOLDER)
        failed += 1
        continue

    md_files = [
        f for f in os.listdir(intro_method_dir)
        if f.endswith(".md")
    ]

    if not md_files:
        print(f"No .md file found in intromethod_output")
        move_to_folder(source_folder, FAILED_FOLDER)
        failed += 1
        continue

    md_path = os.path.join(intro_method_dir, md_files[0])

    with open(md_path, "r", encoding="utf-8") as f:
        paper_content = f.read()

    print(f"Input file: {md_files[0]}")

    extraction_prompt = f"""You are tasked with extracting key information from a research paper for building a knowledge representation.
Paper title: 

Based on the paper content provided below, extract the following information:
- "methods": [List of methods/approaches proposed in the paper],
- "problems": [List of problems the paper addresses],
- "datasets": [List of datasets used for evaluation],
- "metrics": [List of evaluation metrics used],
- "results": [List of objects with 'metric' and 'value' fields representing key quantitative results],
- "novelty_claims": [Claims about what is novel in this work]

Be precise and specific.

Paper content:
{paper_content}

Return ONLY valid JSON in this exact format (no extra text):
{{
  "methods": [...],
  "problems": [...],
  "datasets": [...],
  "metrics": [...],
  "results": [...],
  "novelty_claims": [...]
}}
"""

    max_retries = 5
    response = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": extraction_prompt}],
                temperature=0.0,
                max_tokens=2000
            )
            break

        except RateLimitError:
            wait_time = 30
            print(f"Rate limit hit. Retrying in {wait_time} seconds... ({attempt}/{max_retries})")
            time.sleep(wait_time)

        except Exception as e:
            print(f"Unexpected error for {PAPER_ID}: {e}")
            response = None
            break

    if response is None:
        print(f"Failed API call for {PAPER_ID}")
        move_to_folder(source_folder, FAILED_FOLDER)
        failed += 1
        continue

    raw_content = response.choices[0].message.content.strip()

    try:
        result = json.loads(raw_content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_content, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"JSON parse failed for {PAPER_ID}")
                move_to_folder(source_folder, FAILED_FOLDER)
                failed += 1
                continue
        else:
            print(f"No JSON found for {PAPER_ID}")
            move_to_folder(source_folder, FAILED_FOLDER)
            failed += 1
            continue

    output_file = os.path.join(intro_method_dir, "novelty_claims.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=== EXTRACTED NOVELTY CLAIMS ===\n\n")

        novelty_claims = result.get("novelty_claims", [])
        if novelty_claims:
            for i, claim in enumerate(novelty_claims, 1):
                f.write(f"{i}. {str(claim).strip()}\n")
        else:
            f.write("No novelty claims were extracted.\n")

        f.write("\n\n=== FULL STRUCTURED EXTRACTION (JSON) ===\n")
        f.write(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"Saved: {output_file}")

    moved_to = move_to_folder(source_folder, DONE_FOLDER)
    print(f"Moved to 02_processed: {moved_to}")

    processed += 1

print("=" * 70)
print("Batch processing completed.")
print(f"Successfully processed: {processed}")
print(f"Failed: {failed}")
print(f"Output location: {DONE_FOLDER}")
print("=" * 70)
import os
import json
from openai import OpenAI
import shutil
# ================== CONFIGURATION ==================


BASE_FOLDER = "./06_input"
PROCESSED_FOLDER = "./06_processed"

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# ===================================================


def find_file_pairs(base_folder):
    """
    Find paired *_humanevaluation.json and *_LLMReview.json / LLMReview.json files under:
    BASE_FOLDER / <subfolder> / intromethod_output /
    Skips Windows lock files.
    """
    found = []
    for subfolder in os.listdir(base_folder):
        subfolder_path   = os.path.join(base_folder, subfolder)
        intromethod_path = os.path.join(subfolder_path, "intromethod_output")

        if not os.path.isdir(intromethod_path):
            continue

        human_file = None
        llm_file   = None

        for file in os.listdir(intromethod_path):
            if file.startswith("~$"):
                continue
            if file.endswith("_humanevaluation.json"):
                human_file = os.path.join(intromethod_path, file)
            elif file.endswith("_LLMReview.json") or file == "LLMReview.json":
                llm_file = os.path.join(intromethod_path, file)

        if human_file or llm_file:
            found.append({
                "subfolder":        subfolder,
                "intromethod_path": intromethod_path,
                "human_file":       human_file,
                "llm_file":         llm_file,
            })

    return found


def build_concerns_block_human(evaluation):
    """Pull all comment_excerpts from human evaluation categories."""
    block = ""

    for category in ["Novelty Concern", "Methodology Issues", "Clarity Issues"]:
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


def build_concerns_block_llm(claim_evaluations):
    categories = {
        "Novelty Concern":    [],
        "Methodology Issues": [],
        "Clarity Issues":     [],
        "Others":             [],
    }

    for claim_entry in claim_evaluations:
        if not isinstance(claim_entry, dict):
            continue
        
        evals = claim_entry.get("evaluations", {})
        if not isinstance(evals, dict):
            continue

        for category in ["Novelty Concern", "Methodology Issues", "Clarity Issues"]:
            cat_data = evals.get(category, {})
            if not isinstance(cat_data, dict):
                continue
            explanation = cat_data.get("explanation", "").strip()
            verdict     = cat_data.get("verdict", "")
            if explanation:
                categories[category].append(
                    f"[Verdict: {verdict}] {explanation}"
                )

        others = evals.get("Others", [])
        # Handle case where Others is a dict instead of a list
        if isinstance(others, dict):
            others = [others]
        
        for other in others:
            if not isinstance(other, dict):   # ← THIS was the bug
                continue
            explanation = other.get("explanation", "").strip()
            cat_name    = other.get("category_name", "")
            verdict     = other.get("verdict", "")
            if explanation:
                categories["Others"].append(
                    f"[{cat_name}] [Verdict: {verdict}] {explanation}"
                )

    block = ""
    for category, items in categories.items():
        if not items:
            continue
        block += f"\n[{category.upper()}]\n"
        for item in items:
            block += f"- {item}\n"

    return block.strip()


def count_excerpts(concerns_block):
    """Count number of excerpt lines in the concerns block."""
    return sum(1 for line in concerns_block.splitlines() if line.strip().startswith("-"))


def generate_summary(forum_id, paper_title, concerns_block):
    """Generate a single summary paragraph from a pre-built concerns block."""

    if not concerns_block:
        print("No concern excerpts found")
        return None

    total_excerpts = count_excerpts(concerns_block)
    print(f"Total excerpts to cover: {total_excerpts}")

    system_prompt = (
        "You are an expert academic reviewer. "
        "Respond with ONLY the paragraph, no preamble, no JSON, no extra text."
    )

    user_prompt = f"""You are given structured reviewer concern excerpts for a research paper.
Your task is to write a single overall_summary paragraph.

Forum ID    : {forum_id}
Paper Title : {paper_title}

REVIEWER CONCERN EXCERPTS (grouped by category):
{concerns_block}

There are exactly {total_excerpts} excerpts above. Your paragraph MUST incorporate
specific details from ALL {total_excerpts} of them — do not skip or merge any.

INSTRUCTIONS:
- Write ONE single fluent paragraph synthesizing ALL {total_excerpts} concerns listed above
- Structure the paragraph so it flows from novelty concerns first, then methodology issues,
  then clarity issues, then any other concerns — but do NOT use these as headings or labels,
  just let the narrative naturally transition between them in that order
- Every single excerpt must contribute at least one specific phrase or detail to the paragraph
- Start directly with the most critical specific novelty concern — do NOT open with a generic
  sentence describing what the paper does or what categories are being evaluated
- Do NOT mention the paper name or system name anywhere in the paragraph
- Lift specific phrases, technique names, tool names, and details directly from each excerpt
  verbatim where possible — use the exact words from the excerpts, not paraphrased versions
- Do NOT use any attribution language — no "one reviewer", "another reviewer",
  "a reviewer noted", "reviewers highlighted", or any similar constructions
- No bullet points, no category headers, no sub-sections — just one flowing paragraph
- Every claim must be specific and traceable to something in the excerpts above —
  no vague generalizations without the specific supporting detail from the excerpt
- The paragraph should read like the critical analysis section of an academic paper
- Do NOT stop after covering only the first excerpt — continue until ALL {total_excerpts} are covered
- Do NOT summarize or collapse multiple excerpts into one vague sentence —
  each excerpt deserves its own specific detail in the paragraph"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.0,
            seed=42,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM Error: {e}")
        return None


def read_json(filepath):
    """Read JSON with UTF-8 fallback to latin-1."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return json.load(f)


def process_pair(pair):
    """Generate summaries for both human and LLM review, save into one combined txt."""

    subfolder        = pair["subfolder"]
    intromethod_path = pair["intromethod_path"]
    human_file       = pair["human_file"]
    llm_file         = pair["llm_file"]

    output_path = os.path.join(intromethod_path, f"{subfolder}_REVIEWsummary.txt")

    forum_id      = subfolder
    paper_title   = "N/A"
    human_summary = None
    llm_summary   = None

    # ----------------------------------------------------------------
    # HUMAN EVALUATION SUMMARY
    # ----------------------------------------------------------------
    if human_file:
        print(f"  📄 Reading human evaluation: {os.path.basename(human_file)}")
        try:
            data        = read_json(human_file)
            forum_id    = data.get("forum_id", forum_id)
            paper_title = data.get("paper_title", paper_title)
            evaluation  = data.get("evaluation", {})

            if evaluation:
                concerns_block = build_concerns_block_human(evaluation)
                print("Generating human review summary...")
                human_summary = generate_summary(forum_id, paper_title, concerns_block)
            else:
                print("No evaluation block in human file")
                human_summary = "No evaluation block found."
        except Exception as e:
            print(f"Failed to read human file: {e}")
            human_summary = f"Failed to read file: {e}"
    else:
        print(" No human evaluation file found")
        human_summary = "No human evaluation file found."

    # ----------------------------------------------------------------
    # LLM REVIEW SUMMARY
    # ----------------------------------------------------------------
    if llm_file:
        print(f"Reading LLM review: {os.path.basename(llm_file)}")
        try:
            data        = read_json(llm_file)
            forum_id    = data.get("paper_folder", forum_id)
            claim_evals = data.get("claim_evaluations", [])

            # Ensure claim_evals is a list
            if not isinstance(claim_evals, list):
                print(f"claim_evaluations is not a list (type: {type(claim_evals).__name__}), skipping LLM summary")
                llm_summary = "Invalid claim_evaluations format."
            elif claim_evals:
                concerns_block = build_concerns_block_llm(claim_evals)
                print("Generating LLM review summary...")
                llm_summary = generate_summary(forum_id, paper_title, concerns_block)
            else:
                print("No claim_evaluations block in LLM file")
                llm_summary = "No claim evaluations found."
        except Exception as e:
            print(f"Failed to read LLM file: {e}")
            llm_summary = f"Failed to read file: {e}"
    else:
        print("No LLM review file found")
        llm_summary = "No LLM review file found."

    # ----------------------------------------------------------------
    # SAVE BOTH INTO ONE FILE
    # ----------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Forum ID    : {forum_id}\n")
            f.write(f"Paper Title : {paper_title}\n")
            f.write("=" * 60 + "\n\n")

            f.write("HUMAN REVIEW SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(human_summary or "Summary generation failed.")
            f.write("\n\n")

            f.write("=" * 60 + "\n\n")

            f.write("LLM REVIEW SUMMARY\n")
            f.write("-" * 60 + "\n")
            f.write(llm_summary or "Summary generation failed.")
            f.write("\n")

        print(f"Saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save file: {e}")
        return False


# ================================================================
# MAIN
# ================================================================
print(f"Scanning: {BASE_FOLDER}\n")

pairs = find_file_pairs(BASE_FOLDER)
print(f"Found {len(pairs)} subfolder(s) with evaluation files\n")

if not pairs:
    print("No evaluation files found. Check the folder path.")
    exit(1)

success = 0
failed  = 0

for i, pair in enumerate(pairs, 1):
    print(f"[{i}/{len(pairs)}] {pair['subfolder']}/intromethod_output/")

    ok = process_pair(pair)

    if ok:
        success += 1
    else:
        failed += 1


os.makedirs(PROCESSED_FOLDER, exist_ok=True)

for i, pair in enumerate(pairs, 1):
    print(f"[{i}/{len(pairs)}] {pair['subfolder']}/intromethod_output/")

    ok = process_pair(pair)

    if ok:
        success += 1

        source_folder = os.path.join(BASE_FOLDER, pair["subfolder"])
        target_folder = os.path.join(PROCESSED_FOLDER, pair["subfolder"])

        try:
            if os.path.exists(target_folder):
                shutil.rmtree(target_folder)

            shutil.move(source_folder, target_folder)

            print(f"Moved to processed: {target_folder}")

        except Exception as e:
            print(f"Failed to move folder: {e}")

    else:
        failed += 1

    print()

    print()

print("=" * 60)
print(f"Success : {success}")
print(f"Failed  : {failed}")
print("=" * 60)
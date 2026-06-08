import re
import os
import sys
import shutil
from pathlib import Path
from openai import OpenAI

# ─────────────────────────────────────────────────────────
# API KEY
# ─────────────────────────────────────────────────────────

OPENAI_API_KEY = None
client = OpenAI(api_key=OPENAI_API_KEY or os.getenv("OPENAI_API_KEY"))

if not client.api_key:
    print("Error: No OpenAI API key found.")
    print("Set environment variable: set OPENAI_API_KEY=sk-...")
    sys.exit(1)

MODEL = "gpt-4o"

# ─────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────

BASE_DIR = Path("./01_input")
SUCCESS_DIR = Path("./01_processed")
FAILED_DIR = Path("./01_failed")

MAX_CHARS_FOR_GPT = 450000

# ─────────────────────────────────────────────────────────
# TITLE EXTRACTION
# ─────────────────────────────────────────────────────────

def extract_title_from_md(text):
    if not text.strip():
        return None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r'^#+\s*', '', line)
        line = re.sub(r'\s+', ' ', line)
        m = re.match(r'(.+?[.!?])(?:\s|$)', line)
        if m:
            return m.group(1).strip()
        return line

    return None


# ─────────────────────────────────────────────────────────
# NUMBERED SECTION PARSING
# ─────────────────────────────────────────────────────────

def parse_numbered_headings(text):
    pattern = re.compile(
        r'^(#+)\s+(\d+(?:\.\d+)*)[.\s]+(.+)$',
        re.MULTILINE
    )

    matches = list(pattern.finditer(text))

    if not matches:
        return []

    top_level = min(len(m.group(1)) for m in matches)
    matches = [
        m for m in matches
        if len(m.group(1)) == top_level and '.' not in m.group(2)
    ]

    if not matches:
        return []

    sections = []
    for i, m in enumerate(matches):
        num   = int(m.group(2))
        title = m.group(3).strip()
        start = m.end()
        end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
            "num": num,
            "title": title,
            "start": start,
            "end": end
        })

    return sections


def find_section_by_keywords(sections, keywords):
    for sec in sorted(sections, key=lambda s: s["num"]):
        for kw in keywords:
            if re.search(rf'\b{re.escape(kw)}\b', sec["title"], re.IGNORECASE):
                return sec
    return None


def is_disallowed_as_methods(title):
    disallowed_titles = [
        "Related Work", "Background", "Preliminaries", "Problem Statement",
        "Problem Formulation", "Dataset", "Data", "Experiments", "Experiment",
        "Experimental Setup", "Experimental Results", "Results", "Evaluation",
        "Ablation", "Discussion", "Conclusion", "Conclusions", "Future Work",
        "Limitations", "Appendix", "References"
    ]
    for kw in disallowed_titles:
        if re.search(rf'\b{re.escape(kw)}\b', title, re.IGNORECASE):
            return True
    return False


def is_likely_methods_title(title):
    method_like_titles = [
        "Methodology", "Methods", "Method", "Approach", "Framework",
        "Architecture", "Model", "System", "Algorithm", "Algorithms",
        "Implementation", "Design", "Pipeline", "Procedure",
        "Proposed Method", "Proposed Approach", "Proposed Framework",
        "Our Method", "Our Approach", "System Design"
    ]
    for kw in method_like_titles:
        if re.search(rf'\b{re.escape(kw)}\b', title, re.IGNORECASE):
            return True
    return False


def extract_intro_and_methods_numbered(text, sections):
    intro_kws = ["Introduction"]
    method_kws = [
        "Methodology", "Methods", "Method", "Approach", "Implementation",
        "System Design", "Framework", "Architecture", "Model", "Algorithm",
        "Algorithms", "Pipeline", "Procedure", "Proposed Method",
        "Proposed Approach", "Proposed Framework", "Our Method", "Our Approach"
    ]

    intro_sec  = find_section_by_keywords(sections, intro_kws)
    method_sec = find_section_by_keywords(sections, method_kws)

    intro_content  = None
    method_content = None

    if intro_sec:
        intro_content = text[intro_sec["start"]:intro_sec["end"]].strip()

    if method_sec:
        method_content = text[method_sec["start"]:method_sec["end"]].strip()

    return intro_content, intro_sec, method_content, method_sec


def infer_methods_after_introduction(text, sections, intro_sec):
    if not intro_sec:
        return None, None

    ordered_sections = sorted(sections, key=lambda s: s["num"])

    intro_index = None
    for i, sec in enumerate(ordered_sections):
        if sec["num"] == intro_sec["num"] and sec["title"] == intro_sec["title"]:
            intro_index = i
            break

    if intro_index is None or intro_index + 1 >= len(ordered_sections):
        return None, None

    candidate       = ordered_sections[intro_index + 1]
    candidate_title = candidate["title"].strip()

    if is_disallowed_as_methods(candidate_title):
        return None, None

    candidate_content = text[candidate["start"]:candidate["end"]].strip()

    if not candidate_content:
        return None, None

    return candidate_content, candidate


# ─────────────────────────────────────────────────────────
# KEYWORD FALLBACK
# ─────────────────────────────────────────────────────────

def find_section_start_unnumbered(text, keywords):
    for kw in keywords:
        pat = rf'^(#+)\s*{re.escape(kw)}\b'
        m   = re.search(pat, text, re.I | re.M)
        if m:
            return m.end(), len(m.group(1))
    return None, None


def extract_section_unnumbered(text, start_pos, level, stop_keywords):
    if start_pos is None:
        return None

    stops   = [rf'^#{{1,{level}}}\s+{re.escape(kw)}' for kw in stop_keywords]
    stop_re = "|".join(stops)
    remaining = text[start_pos:]

    m   = re.search(stop_re, remaining, re.I | re.M)
    end = start_pos + m.start() if m else len(text)

    return text[start_pos:end].strip()


# ─────────────────────────────────────────────────────────
# GPT METHODS FALLBACK
# ─────────────────────────────────────────────────────────

def extract_methods_with_llm(text):
    prompt = """
Extract the METHODS or METHODOLOGY section from the markdown paper.

Return exactly:

METHODS_START
[section text]
METHODS_END

If none exists return:
NO_METHOD_SECTION_FOUND
"""
    if len(text) > MAX_CHARS_FOR_GPT:
        print(f" Paper is {len(text):,} chars — truncating to {MAX_CHARS_FOR_GPT:,} for GPT fallback")

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Academic paper parser"},
                {"role": "user", "content": prompt + "\n\n" + text[:MAX_CHARS_FOR_GPT]}
            ],
            temperature=0,
            max_tokens=16000
        )

        content = response.choices[0].message.content

        if "NO_METHOD_SECTION_FOUND" in content:
            return None

        if "METHODS_START" in content:
            return content.split("METHODS_START")[1].split("METHODS_END")[0].strip()

    except Exception as e:
        print(f"OpenAI error: {e}")
        if "context_length_exceeded" in str(e):
            print(f"   → Paper too long even after truncation. Consider splitting the file.")

    return None


# ─────────────────────────────────────────────────────────
# FILE HELPERS
# ─────────────────────────────────────────────────────────

def find_input_md_file(folder, paper_id):
    patterns = [
        f"{paper_id}.md",
        "*paper*.md",
        "*content*.md",
        "*.md"
    ]

    candidates = []
    for p in patterns:
        candidates.extend(folder.glob(p))

    if not candidates:
        return None

    return max(candidates, key=lambda x: x.stat().st_size)


def move_paper_folder(paper_folder: Path, destination_base: Path, reason: str):
    """Move paper folder to destination, handling name conflicts."""
    destination_base.mkdir(parents=True, exist_ok=True)
    dest = destination_base / paper_folder.name

    # If folder already exists at destination, add suffix to avoid overwrite
    if dest.exists():
        dest = destination_base / f"{paper_folder.name}_duplicate"

    try:
        shutil.move(str(paper_folder), str(dest))
        print(f"Moved to {reason}: {dest}")
    except Exception as e:
        print(f"Failed to move folder: {e}")


# ─────────────────────────────────────────────────────────
# PROCESS PAPER
# ─────────────────────────────────────────────────────────

def process_paper(paper_folder):
    paper_id = paper_folder.name
    print(f"\n{'='*60}")
    print(f"Processing: {paper_id}")

    hybrid = paper_folder / "mineru_output" / paper_id / "hybrid_auto"

    if not hybrid.exists():
        print("Missing hybrid_auto folder")
        move_paper_folder(paper_folder, FAILED_DIR, "Couldn't process")
        return False, "missing_folder"

    md_file = find_input_md_file(hybrid, paper_id)

    if not md_file:
        print("No markdown file found")
        move_paper_folder(paper_folder, FAILED_DIR, "Couldn't process")
        return False, "no_md"

    with open(md_file, encoding="utf-8") as f:
        text = f.read()

    # Size diagnostic
    print(f"File size: {len(text):,} chars")
    if len(text) > MAX_CHARS_FOR_GPT:
        print(f"Exceeds GPT limit — regex extraction must succeed or methods will be truncated")

    title         = extract_title_from_md(text)
    intro         = None
    methods       = None
    intro_sec     = None
    method_sec    = None
    method_source = None

    # ── Strategy 1: Numbered headings ──────────────────────
    sections = parse_numbered_headings(text)

    if sections:
        intro, intro_sec, methods, method_sec = extract_intro_and_methods_numbered(text, sections)

        if method_sec:
            method_source = f"numbered heading: {method_sec['title']}"

        if intro_sec and not methods:
            inferred_methods, inferred_method_sec = infer_methods_after_introduction(
                text, sections, intro_sec
            )
            if inferred_methods:
                methods       = inferred_methods
                method_sec    = inferred_method_sec
                method_source = f"inferred after introduction: {inferred_method_sec['title']}"

    # ── Strategy 2: Unnumbered keyword fallback ─────────────
    if not intro:
        pos, lvl = find_section_start_unnumbered(text, ["Introduction"])
        intro = extract_section_unnumbered(
            text, pos, lvl or 1,
            ["Related Work", "Background", "Method", "Methods",
             "Methodology", "Approach", "Framework"]
        )
        if intro:
            print("Introduction found via keyword fallback")

    if not methods:
        pos, lvl = find_section_start_unnumbered(
            text,
            ["Method", "Methods", "Methodology", "Approach",
             "Framework", "Architecture", "Model"]
        )
        methods = extract_section_unnumbered(
            text, pos, lvl or 1,
            ["Results", "Evaluation", "Experiments", "Conclusion",
             "Discussion", "Ablation"]
        )
        if methods:
            method_source = "keyword fallback"
            print("Methods found via keyword fallback")

    # ── Strategy 3: GPT fallback ────────────────────────────
    if not methods:
        print("Using GPT fallback for methods extraction")
        methods = extract_methods_with_llm(text)
        if methods:
            method_source = "GPT fallback"

    # ── Report extraction results ───────────────────────────
    print(f"Title   : {title[:80] if title else 'Not found'}")
    print(f"Intro   : {'Found (' + str(len(intro)) + ' chars)' if intro else 'Not found'}")
    print(f" Methods : {'Found (' + str(len(methods)) + ' chars) via ' + method_source if methods else 'Not found'}")

    # ── If methods not found — move to failed and stop ───────
    if not methods:
        print(f"Methods extraction failed — moving to Couldn't process")
        move_paper_folder(paper_folder, FAILED_DIR, "Couldn't process")
        return False, "no_methods"

    # ── Write output ─────────────────────────────────────────
    output_folder = paper_folder / "intromethod_output"
    output_folder.mkdir(exist_ok=True)

    out_file = output_folder / "intro_and_methods.md"

    parts = []
    parts.append("# Title\n")
    parts.append(title if title else "(Title not found)")
    parts.append("\n\n---\n\n")
    parts.append("# Introduction\n")
    parts.append(intro if intro else "(Introduction not found)")
    parts.append("\n\n---\n\n")
    parts.append("# Methods\n")
    parts.append(methods)

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))

    print(f"Saved: {out_file}")

    # ── Move to success folder ───────────────────────────────
    move_paper_folder(paper_folder, SUCCESS_DIR, "everythingDone")
    return True, "success"


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    if not BASE_DIR.exists():
        print("Base folder not found:", BASE_DIR)
        sys.exit(1)

    folders = [f for f in BASE_DIR.iterdir() if f.is_dir()]

    print(f"Found {len(folders)} paper folder(s) under {BASE_DIR}")

    success = 0
    failed  = 0

    for folder in folders:
        ok, reason = process_paper(folder)
        if ok:
            success += 1
        else:
            failed += 1

    print(f"\n{'='*60}")
    print(f"Completed  : {success}  →  {SUCCESS_DIR}")
    print(f"Failed     : {failed}   →  {FAILED_DIR}")
    print(f"Total      : {len(folders)}")


if __name__ == "__main__":
    main()
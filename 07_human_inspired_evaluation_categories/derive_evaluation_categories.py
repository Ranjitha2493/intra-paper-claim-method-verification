import os
import json
import re
import pandas as pd
from collections import Counter
from openai import OpenAI

# ============================================================
# CONFIGURATION
# ============================================================

REVIEW_FOLDER = "./input_reviews"

OUTPUT_DIR = "./output"
OUTPUT_EXCEL = os.path.join(
    OUTPUT_DIR,
    "review_concern_analysis.xlsx"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = "gpt-4o"

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)

# ============================================================
# GPT PROMPTS
# ============================================================

SYSTEM_PROMPT = """
You are an expert analyst of academic peer reviews.

Your task is to identify reviewer concerns.

IMPORTANT:

- Do NOT use predefined categories.
- Do NOT invent concerns.
- Extract only concerns explicitly raised by reviewers.
- Create a short concern label (2-6 words).
- Preserve reviewer intent.
- Multiple concerns may exist.

Return ONLY valid JSON.

Format:

{
  "concerns":[
    {
      "concern_label":"",
      "evidence":""
    }
  ]
}
"""

# ============================================================
# HELPERS
# ============================================================

def clean_json_response(text):

    text = text.strip()

    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)

    return text.strip()


def read_json(filepath):

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)

    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return json.load(f)


def sanitize_string(value):
    """Remove control characters that openpyxl cannot write to Excel."""
    if not isinstance(value, str):
        return value
    # Remove control characters (0x00-0x1F except tab, newline, carriage return)
    return "".join(
        char if ord(char) >= 32 or char in "\t\n\r" else ""
        for char in value
    )


def sanitize_dataframe(df):
    """Sanitize all string columns in a dataframe."""
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(sanitize_string)
    return df


def find_review_files(base_folder):

    review_files = []

    for paper_id in os.listdir(base_folder):

        paper_folder = os.path.join(
            base_folder,
            paper_id
        )

        if not os.path.isdir(paper_folder):
            continue

        json_file = os.path.join(
            paper_folder,
            f"{paper_id}.json"
        )

        if os.path.exists(json_file):
            review_files.append(json_file)

    return review_files


def extract_review_text(review):

    content = review.get("content", {})

    sections = []

    weaknesses = content.get("weaknesses", "")
    questions = content.get("questions", "")

    if weaknesses:
        sections.append(
            f"WEAKNESSES:\n{weaknesses}"
        )

    if questions:
        sections.append(
            f"QUESTIONS:\n{questions}"
        )

    return "\n\n".join(sections)


# ============================================================
# GPT EXTRACTION
# ============================================================

def extract_concerns(review_text):

    prompt = f"""
Review Text:

{review_text}

Identify every distinct reviewer concern.

Return JSON only.
"""

    try:

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = (
            response
            .choices[0]
            .message.content
        )

        content = clean_json_response(content)

        return json.loads(content)

    except Exception as e:

        print("GPT Error:", e)

        return {
            "concerns": []
        }


# ============================================================
# MAIN PROCESSING
# ============================================================

print("Loading review files...")

review_files = find_review_files(
    REVIEW_FOLDER
)

print(
    f"Found {len(review_files)} papers"
)

all_concerns = []

for file_index, review_file in enumerate(
    review_files,
    start=1
):

    print(
        f"[{file_index}/{len(review_files)}] "
        f"{os.path.basename(review_file)}"
    )

    try:

        paper_data = read_json(
            review_file
        )

        forum_id = paper_data.get(
            "forum_id",
            ""
        )

        paper_title = paper_data.get(
            "paper_title",
            ""
        )

        reviews = paper_data.get(
            "reviews",
            []
        )

        for review in reviews:

            review_id = review.get(
                "review_id",
                ""
            )

            review_text = (
                extract_review_text(
                    review
                )
            )

            if not review_text.strip():
                continue

            result = extract_concerns(
                review_text
            )

            concerns = result.get(
                "concerns",
                []
            )

            for concern in concerns:

                all_concerns.append({
                    "forum_id":
                        forum_id,

                    "paper_title":
                        paper_title,

                    "review_id":
                        review_id,

                    "concern_label":
                        concern.get(
                            "concern_label",
                            ""
                        ),

                    "evidence":
                        concern.get(
                            "evidence",
                            ""
                        )
                })

    except Exception as e:

        print(
            f"Error processing "
            f"{review_file}"
        )

        print(e)

# ============================================================
# DATAFRAME
# ============================================================

concerns_df = pd.DataFrame(
    all_concerns
)

if concerns_df.empty:

    print(
        "No concerns extracted."
    )

    raise SystemExit

# ============================================================
# FREQUENCY TABLE
# ============================================================

frequency_df = (
    concerns_df
    .groupby("concern_label")
    .size()
    .reset_index(name="count")
    .sort_values(
        by="count",
        ascending=False
    )
)

# ============================================================
# SAMPLE CONCERNS
# ============================================================

sample_df = (
    concerns_df
    .groupby("concern_label")
    .head(10)
)

# ============================================================
# PAPER LEVEL SUMMARY
# ============================================================

paper_summary_df = (
    concerns_df
    .groupby(
        [
            "forum_id",
            "paper_title"
        ]
    )
    ["concern_label"]
    .apply(
        lambda x:
        "; ".join(
            sorted(
                set(x)
            )
        )
    )
    .reset_index()
)

# ============================================================
# DISCOVER COMMON THEMES
# ============================================================

top_labels = (
    frequency_df
    .head(100)
    ["concern_label"]
    .tolist()
)

theme_prompt = f"""
Below are reviewer concern labels extracted
from many peer reviews.

Group similar concerns together and identify
higher-level recurring themes.

Concern Labels:

{json.dumps(top_labels, indent=2)}

Return JSON:

{{
  "themes":[
    {{
      "theme":"",
      "related_labels":[]
    }}
  ]
}}
"""

try:

    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role":"system",
                "content":
                "You are an expert meta-review analyst."
            },
            {
                "role":"user",
                "content":
                theme_prompt
            }
        ]
    )

    themes_json = json.loads(
        clean_json_response(
            response.choices[0]
            .message.content
        )
    )

    theme_rows = []

    for theme in themes_json.get(
        "themes",
        []
    ):

        theme_name = theme.get(
            "theme",
            ""
        )

        for label in theme.get(
            "related_labels",
            []
        ):

            theme_rows.append({
                "theme":
                    theme_name,
                "concern_label":
                    label
            })

    themes_df = pd.DataFrame(
        theme_rows
    )

except Exception as e:

    print(
        "Theme generation failed:",
        e
    )

    themes_df = pd.DataFrame()

# ============================================================
# SAVE EXCEL
# ============================================================

# Sanitize all dataframes before writing to Excel
concerns_df = sanitize_dataframe(concerns_df.copy())
frequency_df = sanitize_dataframe(frequency_df.copy())
sample_df = sanitize_dataframe(sample_df.copy())
paper_summary_df = sanitize_dataframe(paper_summary_df.copy())
themes_df = sanitize_dataframe(themes_df.copy())

with pd.ExcelWriter(
    OUTPUT_EXCEL,
    engine="openpyxl"
) as writer:

    concerns_df.to_excel(
        writer,
        sheet_name="ExtractedConcerns",
        index=False
    )

    frequency_df.to_excel(
        writer,
        sheet_name="ConcernFrequency",
        index=False
    )

    sample_df.to_excel(
        writer,
        sheet_name="SampleConcerns",
        index=False
    )

    paper_summary_df.to_excel(
        writer,
        sheet_name="PaperSummary",
        index=False
    )

    themes_df.to_excel(
        writer,
        sheet_name="DiscoveredThemes",
        index=False
    )

print()
print("=" * 70)
print("Analysis Complete")
print(f"Papers processed : {concerns_df['forum_id'].nunique()}")
print(f"Reviews processed: {concerns_df['review_id'].nunique()}")
print(f"Concerns found   : {len(concerns_df)}")
print(f"Saved report     : {OUTPUT_EXCEL}")
print("=" * 70)


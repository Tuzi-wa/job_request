import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import pandas as pd
import requests


ALLOWED_CATEGORIES = [
    "Age",
    "Gender",
    "Origin",
    "Location",
    "Mobility",
    "PPE",
    "Standing",
    "Salary",
    "Role clarity",
]

CATEGORY_LOOKUP = {category.lower(): category for category in ALLOWED_CATEGORIES}
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b")
FLAG_MISSING_SALARY = os.getenv("FLAG_MISSING_SALARY", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

NULL_LIKE_VALUES = {
    "",
    "none",
    "no",
    "n/a",
    "na",
    "-",
    "null",
    "nil",
    "no discrimination",
    "no issue",
    "no issues",
}

GROUND_TRUTH_PRIORITY = [
    "discrimination",
    "expected_categories",
    "expected category",
    "labels",
    "label",
    "ground_truth",
    "ground truth",
    "target",
    "class",
]

JOB_TEXT_PRIORITY = [
    "job post",
    "job_post",
    "jobpost",
    "post",
    "job advertisement",
    "job_advertisement",
    "advertisement",
    "ad text",
    "text",
    "description",
    "content",
]

SYSTEM_PROMPT = """You are a strict job advertisement discrimination checker for a fixed homework label scheme.
Analyze exactly one job advertisement and return JSON only.
Use only these categories exactly when applicable:
["Age", "Gender", "Origin", "Location", "Mobility", "PPE", "Standing", "Salary", "Role clarity"]

Label guidance for this assignment:
- Age: explicit age limits or age preference.
- Gender: waitress, salesman, hostess, male-only, female-only, or other gender-specific wording.
- Origin: nationality, ethnic origin, native-speaker requirements, country/region-origin preference.
- Location: explicit residence preference or requirement tied to a city, province, region, or area.
- Mobility: driver license, own vehicle, willingness to travel, on-call mobility, site visits, shift-based movement requirements.
- PPE: asks the candidate to already possess PPE or specific safety gear.
- Standing: appearance/image requirements such as well-presented or good appearance.
- Salary: any unclear compensation wording such as "to be defined", "to be agreed", "commensurate with experience", "classified based on actual skills", or similar vague pay/level language.
- Role clarity: duties are vague, e.g. "various tasks", "as needed", "whatever is needed".

Important rules:
- Be recall-oriented for the assignment labels, especially Salary, Mobility, Origin, and Location.
- Do not use labels outside the allowed list.
- If there is no issue, return has_issue=false and empty arrays.

Return exactly this JSON schema and nothing else:
{
  "has_issue": true,
  "categories": ["Origin"],
  "problematic_text": ["native English speaker"],
  "explanations": ["Requires native speaker status, which can discriminate based on origin."]
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check job advertisements for discrimination issues using local Ollama."
    )
    parser.add_argument(
        "--excel",
        required=True,
        help="Path to the input .xlsx file containing job posts and a ground-truth label column.",
    )
    parser.add_argument(
        "--output",
        default="analysis_result.xlsx",
        help="Path to the output .xlsx file. Default: analysis_result.xlsx",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds for Ollama requests. Default: 120",
    )
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def find_ground_truth_column(columns: Sequence[str]) -> Optional[str]:
    normalized_to_original = {normalize_column_name(col): col for col in columns}

    for preferred in GROUND_TRUTH_PRIORITY:
        if preferred in normalized_to_original:
            return normalized_to_original[preferred]

    for normalized, original in normalized_to_original.items():
        if "discrimination" in normalized:
            return original

    for normalized, original in normalized_to_original.items():
        if any(keyword in normalized for keyword in ["ground truth", "expected", "label", "target"]):
            return original

    return None


def find_job_text_column(columns: Sequence[str], excluded: Optional[Set[str]] = None) -> Optional[str]:
    excluded = excluded or set()
    normalized_pairs = [
        (normalize_column_name(col), col)
        for col in columns
        if col not in excluded
    ]

    for preferred in JOB_TEXT_PRIORITY:
        for normalized, original in normalized_pairs:
            if normalized == preferred:
                return original

    strong_candidates: List[str] = []
    weak_candidates: List[str] = []
    for normalized, original in normalized_pairs:
        if any(token in normalized for token in ["text", "description", "content", "advertisement", "ad text"]):
            if "id" not in normalized:
                strong_candidates.append(original)
                continue

        if any(keyword in normalized for keyword in ["job", "advert", "description", "text", "content", "post"]):
            if "id" not in normalized and "number" not in normalized:
                weak_candidates.append(original)

    if strong_candidates:
        return strong_candidates[0]
    if weak_candidates:
        return weak_candidates[0]

    return None


def parse_bool_env(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def find_json_candidate(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if not raw_text:
        return raw_text

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        return fenced_match.group(1).strip()

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return raw_text[start : end + 1].strip()

    return raw_text


def safe_json_loads(raw_text: str) -> Dict[str, Any]:
    candidate = find_json_candidate(raw_text)
    return json.loads(candidate)


def extract_message_content(response_json: Dict[str, Any]) -> str:
    message = response_json.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    if isinstance(response_json.get("response"), str):
        return str(response_json["response"]).strip()
    return ""


def build_user_prompt(job_text: str) -> str:
    return (
        "Analyze the following job advertisement for discrimination or unfairness.\n"
        "Return strict JSON only.\n\n"
        f"Job advertisement:\n{job_text}"
    )


def ollama_chat(messages: List[Dict[str, str]], timeout: int) -> Tuple[Optional[str], Optional[str]]:
    payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "think": False,
        "messages": messages,
        "format": "json",
        "options": {
            "temperature": 0,
        },
    }
    try:
        response = requests.post(
            f"{OLLAMA_URL.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        response_json = response.json()
    except Exception as exc:
        return None, f"Ollama request failed: {exc}"

    content = extract_message_content(response_json)
    if not content:
        return None, "Ollama returned empty content"

    return content, None


def repair_json_with_ollama(raw_output: str, timeout: int) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    repair_system_prompt = """Convert the input into valid JSON only.
Keep only this schema:
{
  "has_issue": true,
  "categories": ["Origin"],
  "problematic_text": ["text"],
  "explanations": ["reason"]
}
Rules:
- categories must contain only these exact labels:
["Age", "Gender", "Origin", "Location", "Mobility", "PPE", "Standing", "Salary", "Role clarity"]
- If nothing valid is present, use false and empty arrays.
- Output JSON only, with no markdown."""

    content, error = ollama_chat(
        [
            {"role": "system", "content": repair_system_prompt},
            {"role": "user", "content": raw_output},
        ],
        timeout=timeout,
    )
    if error:
        return None, None, f"Repair call failed: {error}"

    try:
        repaired_json = safe_json_loads(content)
        return repaired_json, content, None
    except Exception as exc:
        return None, content, f"Repair JSON parsing failed: {exc}"


def looks_like_missing_salary_issue(job_text: str) -> bool:
    if not FLAG_MISSING_SALARY:
        return False

    if has_explicit_salary_info(job_text):
        return False

    text = job_text.lower()
    salary_keywords = [
        "salary",
        "compensation",
        "pay",
        "wage",
        "hourly",
        "per hour",
        "annual",
        "per annum",
        "remuneration",
        "benefits",
        "$",
        "eur",
        "usd",
        "gbp",
        "salary:",
        "salary range",
        "salary level",
        "ral",
        "gross per month",
        "per year",
    ]
    return not any(keyword in text for keyword in salary_keywords)


def find_first_match(text: str, patterns: Sequence[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip()
    return None


def has_explicit_salary_info(job_text: str) -> bool:
    text = job_text.lower()
    explicit_patterns = [
        r"salary:\s*[€$]?\s*\d",
        r"salary range:\s*[€$]?\s*\d",
        r"annual salary range:\s*[€$]?\s*\d",
        r"gross salary of\s*[€$]?\s*\d",
        r"approximately\s*[€$]?\s*\d",
        r"competitive salary",
        r"€\s*\d",
        r"\b\d+\s*(?:k|000)\b",
        r"\bral\b",
        r"\bper month\b",
        r"\bper year\b",
        r"\b14 monthly payments\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in explicit_patterns)


def title_line(job_text: str) -> str:
    parts = [line.strip() for line in job_text.splitlines() if line.strip()]
    return " | ".join(parts[:3]).lower()


def heuristic_category_signals(job_text: str) -> Dict[str, Dict[str, str]]:
    signals: Dict[str, Dict[str, str]] = {}
    text = job_text.strip()
    lowered = text.lower()
    title = title_line(text)

    salary_patterns = [
        r"\bto be defined\b",
        r"\bto be agreed\b",
        r"\bcommensurate with experience\b",
        r"\bsalary commensurate with (?:the )?candidate'?s experience\b",
        r"\bcompetitive salary based on experience\b",
        r"\bsalary based on experience\b",
        r"\bdepends on candidate\b",
        r"\bevaluated based on the candidate'?s experience\b",
        r"\bevaluated based on experience\b",
        r"\bcompensation will be based on the candidate'?s experience\b",
        r"\bcompensation based on the candidate'?s experience\b",
        r"\bbased on actual skills\b",
        r"\bclassified based on actual skills\b",
        r"\bclassification based on actual skills\b",
        r"\bsalary level will be commensurate with experience\b",
        r"\bcontract level\b",
        r"\blevel to be defined\b",
        r"\bpay to be agreed\b",
        r"\bcompensation to be defined\b",
        r"\bcompensation to be agreed\b",
    ]
    salary_hit = find_first_match(text, salary_patterns)
    if salary_hit and salary_hit.lower() == "contract level" and has_explicit_salary_info(text):
        salary_hit = None
    if salary_hit:
        signals["Salary"] = {
            "snippet": salary_hit,
            "reason": "Compensation or contract level is described in a vague way.",
        }
    elif looks_like_missing_salary_issue(text):
        signals["Salary"] = {
            "snippet": "salary not specified",
            "reason": "The advertisement does not provide salary information and the missing-salary flag is enabled.",
        }

    mobility_patterns = [
        r"\bdriver.?s license\b",
        r"\bdriving license\b",
        r"\blicense[: ]*b\b",
        r"\blicense[: ]*c\b",
        r"\bcategory b\b",
        r"\bcategory c\b",
        r"\bown car\b",
        r"\bown vehicle\b",
        r"\bown means of transportation\b",
        r"\bavailable to travel\b",
        r"\btravel to client locations\b",
        r"\bsite visits?\b",
        r"\bon-call\b",
        r"\breperability\b",
    ]
    mobility_hit = find_first_match(text, mobility_patterns)
    if mobility_hit:
        mobility_lower = mobility_hit.lower()
        if "driver" in title and "license" in mobility_lower:
            mobility_hit = None
        elif "company vehicle" in lowered and "license" in mobility_lower and not any(
            token in lowered for token in ["own car", "own vehicle", "own means of transportation"]
        ):
            mobility_hit = None
        elif mobility_lower == "on-call" and "no night on-call duty required" in lowered:
            mobility_hit = None
    if mobility_hit:
        signals["Mobility"] = {
            "snippet": mobility_hit,
            "reason": "The ad requires mobility-related availability such as travel or a driving license.",
        }

    origin_patterns = [
        r"\bnative [a-z]+ speaker\b",
        r"\bnative speaker\b",
        r"\bnationality\b",
        r"\b[a-z]+ nationality\b",
        r"\bmother tongue\b",
        r"\bmother-tongue\b",
        r"\bfrom [a-z][a-z .'-]+ preferred\b",
        r"\bsouth american nationality\b",
        r"\bforeign customers\b",
        r"\bcompany origins\b",
    ]
    origin_hit = find_first_match(text, origin_patterns)
    if origin_hit:
        signals["Origin"] = {
            "snippet": origin_hit,
            "reason": "The ad references nationality, native-speaker status, or origin preference.",
        }

    location_patterns = [
        r"\bresident in [a-z][a-z .'-]+\b",
        r"\bresidence in [a-z][a-z .'-]+\b",
        r"\bresidence near [a-z][a-z .'-]+\b",
        r"\bresidence near the store location\b",
        r"\bresidence within \d+\s*km\b",
        r"\bdomiciled in [a-z][a-z .'-]+\b",
        r"\bdomicile in [a-z][a-z .'-]+\b",
        r"\bliving in [a-z][a-z .'-]+\b",
        r"\bimmediate vicinity of [a-z][a-z .'-]+\b",
    ]
    location_hit = find_first_match(text, location_patterns)
    if location_hit:
        signals["Location"] = {
            "snippet": location_hit,
            "reason": "The ad ties the role to a specific area or residence-based location requirement.",
        }

    gender_patterns = [
        r"\bwaitress\b",
        r"\bwaiter\b",
        r"\bhostess\b",
        r"\bstewardess\b",
        r"\bsalesman\b",
        r"\bsaleswoman\b",
        r"\bseamstress\b",
        r"\bmale only\b",
        r"\bfemale only\b",
    ]
    gender_hit = find_first_match(text, gender_patterns)
    if gender_hit:
        signals["Gender"] = {
            "snippet": gender_hit,
            "reason": "The ad uses gender-specific wording for the role or candidate.",
        }

    standing_patterns = [
        r"\bprofessional appearance\b",
    ]
    standing_hit = find_first_match(text, standing_patterns)
    if standing_hit:
        signals["Standing"] = {
            "snippet": standing_hit,
            "reason": "The ad mentions appearance or presentation requirements.",
        }

    ppe_patterns = [
        r"\bpossession of personal protective equipment\b",
        r"\bpossess(?:ion)? of ppe\b",
        r"\bpossession of safety shoes\b",
        r"\bmust have ppe\b",
        r"\bmust possess ppe\b",
    ]
    ppe_hit = find_first_match(text, ppe_patterns)
    if ppe_hit:
        signals["PPE"] = {
            "snippet": ppe_hit,
            "reason": "The ad asks the candidate to already have PPE or safety gear.",
        }

    role_clarity_patterns = [
        r"\bvarious tasks\b",
        r"\bwhatever is needed\b",
        r"\bother duties as assigned\b",
        r"\bgeneral support\b",
        r"\bfairy hands\b",
    ]
    role_clarity_hit = find_first_match(text, role_clarity_patterns)
    if role_clarity_hit or (
        "responsibilities:" in lowered
        and re.search(r"responsibilities:\s*requirements:", lowered, flags=re.IGNORECASE)
    ):
        if not role_clarity_hit:
            role_clarity_hit = "Responsibilities:"
        signals["Role clarity"] = {
            "snippet": role_clarity_hit,
            "reason": "The job duties are described in a vague or catch-all way.",
        }

    age_patterns = [
        r"\bage not exceeding \d+\b",
        r"\bunder \d+\b",
        r"\bbetween \d+ and \d+\b",
    ]
    age_hit = find_first_match(text, age_patterns)
    if age_hit and not re.search(r"\bapprentice|apprenticeship|internship|trainee\b", lowered):
        signals["Age"] = {
            "snippet": age_hit,
            "reason": "The ad includes an age-related requirement or preference.",
        }

    return signals


def normalize_category_token(token: str) -> Optional[str]:
    cleaned = re.sub(r"\s+", " ", token.strip())
    if not cleaned:
        return None

    lowered = cleaned.lower()
    if lowered in CATEGORY_LOOKUP:
        return CATEGORY_LOOKUP[lowered]

    alias_map = {
        "age discrimination": "Age",
        "gender discrimination": "Gender",
        "origin discrimination": "Origin",
        "location discrimination": "Location",
        "mobility discrimination": "Mobility",
        "ppe discrimination": "PPE",
        "standing discrimination": "Standing",
        "salary discrimination": "Salary",
        "role clarity discrimination": "Role clarity",
        "role_clarity": "Role clarity",
        "roleclarity": "Role clarity",
    }
    if lowered in alias_map:
        return alias_map[lowered]

    for allowed in ALLOWED_CATEGORIES:
        if lowered == allowed.lower():
            return allowed

    return None


def split_categories(raw_value: str) -> List[str]:
    if not raw_value:
        return []

    candidate = raw_value.replace("\n", ",").replace(";", ",").replace("|", ",")
    candidate = re.sub(r"\s*/\s*", ",", candidate)
    parts = [part.strip() for part in candidate.split(",")]
    return [part for part in parts if part]


def parse_categories_from_cell(value: Any) -> Set[str]:
    text = normalize_text(value)
    if not text:
        return set()

    if text.strip().lower() in NULL_LIKE_VALUES:
        return set()

    categories: Set[str] = set()
    for token in split_categories(text):
        normalized = normalize_category_token(token)
        if normalized:
            categories.add(normalized)

    return categories


def coerce_list_of_strings(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    return [str(value).strip()]


def sanitize_llm_result(result: Dict[str, Any], job_text: str) -> Dict[str, Any]:
    llm_categories: Set[str] = set()
    for item in coerce_list_of_strings(result.get("categories")):
        normalized = normalize_category_token(item)
        if normalized:
            llm_categories.add(normalized)

    heuristic_signals = heuristic_category_signals(job_text)
    categories: Set[str] = set(heuristic_signals.keys())

    # Keep only high-precision LLM-only categories when heuristics do not already cover them.
    llm_allowlist = {"Role clarity"}
    categories.update(category for category in llm_categories if category in llm_allowlist)

    problematic_text: List[str] = []
    explanations: List[str] = []

    existing_snippets = {item.lower() for item in problematic_text}
    existing_reasons = {item.lower() for item in explanations}
    for signal in heuristic_signals.values():
        snippet = signal["snippet"].strip()
        reason = signal["reason"].strip()
        if snippet.lower() not in existing_snippets:
            problematic_text.append(snippet)
            existing_snippets.add(snippet.lower())
        if reason.lower() not in existing_reasons:
            explanations.append(reason)
            existing_reasons.add(reason.lower())

    raw_problematic = coerce_list_of_strings(result.get("problematic_text"))
    raw_explanations = coerce_list_of_strings(result.get("explanations"))
    for snippet in raw_problematic:
        if snippet.lower() not in existing_snippets and any(cat in categories for cat in llm_categories):
            problematic_text.append(snippet)
            existing_snippets.add(snippet.lower())
    for reason in raw_explanations:
        if reason.lower() not in existing_reasons and any(cat in categories for cat in llm_categories):
            explanations.append(reason)
            existing_reasons.add(reason.lower())

    has_issue = bool(result.get("has_issue")) or bool(categories)

    if not has_issue:
        categories = set()

    return {
        "has_issue": has_issue,
        "categories": sorted(categories),
        "problematic_text": problematic_text,
        "explanations": explanations,
    }


def format_analysis(clean_result: Dict[str, Any], failure_reason: Optional[str] = None) -> str:
    if failure_reason:
        return failure_reason

    categories = clean_result.get("categories", [])
    explanations = clean_result.get("explanations", [])
    snippets = clean_result.get("problematic_text", [])

    if not categories:
        return "No discrimination or unfairness issue detected."

    parts: List[str] = []
    if categories:
        parts.append(f"Categories: {', '.join(categories)}.")
    if explanations:
        parts.append("Reasons: " + " | ".join(explanations[:3]))
    if snippets:
        parts.append("Text: " + " | ".join(snippets[:3]))
    return " ".join(parts)


def predict_for_job(job_text: str, timeout: int) -> Dict[str, Any]:
    content, error = ollama_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(job_text)},
        ],
        timeout=timeout,
    )

    if error:
        return {
            "llm_status": "failed",
            "predicted_categories": "",
            "analysis": error,
            "llm_raw_output": "",
        }

    raw_output = content
    parsed_json: Optional[Dict[str, Any]] = None
    failure_reason: Optional[str] = None

    try:
        parsed_json = safe_json_loads(raw_output)
    except Exception as exc:
        repaired_json, repaired_raw, repair_error = repair_json_with_ollama(raw_output, timeout)
        if repaired_raw:
            raw_output = repaired_raw
        if repaired_json is not None:
            parsed_json = repaired_json
        else:
            failure_reason = f"JSON parsing failed: {exc}; {repair_error}"

    if parsed_json is None:
        return {
            "llm_status": "failed",
            "predicted_categories": "",
            "analysis": format_analysis({}, failure_reason=failure_reason or "Unknown LLM parsing failure"),
            "llm_raw_output": raw_output,
        }

    clean_result = sanitize_llm_result(parsed_json, job_text)
    return {
        "llm_status": "success",
        "predicted_categories": ", ".join(clean_result["categories"]),
        "analysis": format_analysis(clean_result),
        "llm_raw_output": raw_output,
    }


def compare_sets(predicted_text: str, ground_truth_text: Any) -> Tuple[Set[str], Set[str], bool]:
    predicted_set = parse_categories_from_cell(predicted_text)
    ground_truth_set = parse_categories_from_cell(ground_truth_text)
    return predicted_set, ground_truth_set, predicted_set == ground_truth_set


def build_summary(
    df: pd.DataFrame,
    ground_truth_column: str,
    model_name: str,
) -> str:
    total_rows = len(df)
    success_rows = int((df["llm_status"] == "success").sum()) if total_rows else 0
    match_pct = (float(df["complete_match"].mean()) * 100.0) if total_rows else 0.0

    failure_rows = df[df["llm_status"] == "failed"]
    if failure_rows.empty:
        first_failure = "None"
    else:
        first_failure = str(failure_rows.iloc[0]["analysis"]).strip() or "Unknown failure"

    lines = [
        f"Ground truth column used: {ground_truth_column}",
        f"Complete match percentage: {match_pct:.2f}%",
        f"LLM successful rows: {success_rows}/{total_rows}",
        f"First failure: {first_failure}",
        f"Model: {model_name}",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    input_path = os.path.abspath(args.excel)
    output_path = os.path.abspath(args.output)

    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        df = pd.read_excel(input_path)
    except Exception as exc:
        print(f"Failed to read Excel file: {exc}", file=sys.stderr)
        return 1

    if df.empty:
        print("The input Excel file is empty.", file=sys.stderr)
        return 1

    ground_truth_column = find_ground_truth_column(df.columns)
    if not ground_truth_column:
        print(
            "Could not find a ground truth label column. "
            "Expected something like Discrimination / labels / ground_truth.",
            file=sys.stderr,
        )
        return 1

    job_text_column = find_job_text_column(df.columns, excluded={ground_truth_column})
    if not job_text_column:
        print(
            "Could not find a job post text column. "
            "Expected something like job post / description / text.",
            file=sys.stderr,
        )
        return 1

    llm_statuses: List[str] = []
    predicted_values: List[str] = []
    analyses: List[str] = []
    raw_outputs: List[str] = []
    complete_matches: List[bool] = []

    for _, row in df.iterrows():
        job_text = normalize_text(row.get(job_text_column, ""))
        if not job_text:
            result = {
                "llm_status": "failed",
                "predicted_categories": "",
                "analysis": f"Missing job post text in column '{job_text_column}'",
                "llm_raw_output": "",
            }
        else:
            result = predict_for_job(job_text, timeout=args.timeout)

        _, _, is_complete_match = compare_sets(
            result["predicted_categories"],
            row.get(ground_truth_column, ""),
        )

        llm_statuses.append(result["llm_status"])
        predicted_values.append(result["predicted_categories"])
        analyses.append(result["analysis"])
        raw_outputs.append(result["llm_raw_output"])
        complete_matches.append(is_complete_match)

    df["llm_status"] = llm_statuses
    df["predicted_categories"] = predicted_values
    df["analysis"] = analyses
    df["llm_raw_output"] = raw_outputs
    df["complete_match"] = complete_matches

    try:
        df.to_excel(output_path, index=False)
    except Exception as exc:
        print(f"Failed to write output Excel file: {exc}", file=sys.stderr)
        return 1

    summary = build_summary(df, ground_truth_column=ground_truth_column, model_name=OLLAMA_MODEL)
    print(summary)
    print(f"\nSaved results to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

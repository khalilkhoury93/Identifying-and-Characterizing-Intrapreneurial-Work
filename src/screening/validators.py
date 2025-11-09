"""Validators for LLM screening responses."""

import json
import re
from typing import Any


def parse_json_strict(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Extract and parse the first JSON object from text.

    This handles cases where LLMs include markdown code blocks or extra text.

    Args:
        text: Raw text that should contain JSON

    Returns:
        Tuple of (parsed_dict, errors)
        - parsed_dict is None if parsing failed
        - errors is list of error messages
    """
    errors = []

    # Try direct parse first
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, dict):
            return obj, []
        else:
            errors.append(f"JSON parsed but got {type(obj).__name__}, expected dict")
    except json.JSONDecodeError:
        pass  # Try extraction methods below

    # Try to extract JSON from markdown code blocks
    # Pattern: ```json\n{...}\n```
    markdown_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
    matches = re.findall(markdown_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    return obj, []
            except json.JSONDecodeError:
                continue

    # Try to find first {...} block
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    matches = re.findall(brace_pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            try:
                obj = json.loads(match)
                if isinstance(obj, dict):
                    return obj, []
            except json.JSONDecodeError:
                continue

    # All methods failed
    errors.append("Could not extract valid JSON dict from response")
    return None, errors


def validate_schema(
    obj: dict[str, Any],
    required_fields: list[str],
    optional_fields: list[str] | None = None,
) -> tuple[bool, list[str]]:
    """Validate that JSON object has required fields with correct types.

    Expected schema for entrepreneurial_filter.v2/v2.1:
    {
        "entrepreneurial": int (0 or 1),
        "confidence": float (0.0 to 1.0),
        "evidence_phrases": list[str] (max 3),
        "subtype": str or null,
        "tags": dict (optional),
        "uncertain": bool (optional, v2.1)
    }

    Args:
        obj: Parsed JSON object
        required_fields: List of required field names
        optional_fields: List of optional field names

    Returns:
        Tuple of (is_valid, errors)
    """
    errors = []
    
    # v2.1 adds 'uncertain' as optional field
    if optional_fields is None:
        optional_fields = []
    if "uncertain" not in optional_fields:
        optional_fields = list(optional_fields) + ["uncertain"]

    # Check required fields exist
    for field in required_fields:
        if field not in obj:
            errors.append(f"Missing required field: {field}")

    # Validate field types and constraints
    if "entrepreneurial" in obj:
        val = obj["entrepreneurial"]
        if not isinstance(val, int) or val not in (0, 1):
            errors.append(f"'entrepreneurial' must be 0 or 1, got {val}")

    if "confidence" in obj:
        val = obj["confidence"]
        # Accept either numeric [0.0,1.0] or categorical {High, Medium, Low}
        if isinstance(val, (int, float)):
            if not (0.0 <= float(val) <= 1.0):
                errors.append(f"'confidence' numeric must be in [0.0, 1.0], got {val}")
            else:
                # Normalize numeric confidence into categorical string for downstream consistency
                num = float(val)
                label = "High" if num >= 0.8 else ("Medium" if num >= 0.6 else "Low")
                obj["confidence"] = label
        elif isinstance(val, str):
            label = val.strip()
            allowed = {"High", "Medium", "Low"}
            # Accept case-insensitive variants by normalizing to title case
            label_tc = label.title()
            if label_tc not in allowed:
                errors.append(
                    f"'confidence' must be one of {sorted(allowed)} or a numeric in [0,1], got '{val}'"
                )
            else:
                obj["confidence"] = label_tc
        else:
            errors.append(
                f"'confidence' must be a string in {['High','Medium','Low']} or numeric in [0,1], got {type(val).__name__}"
            )

    if "evidence_phrases" in obj:
        val = obj["evidence_phrases"]
        if not isinstance(val, list):
            errors.append(
                f"'evidence_phrases' must be a list, got {type(val).__name__}"
            )
        elif not all(isinstance(x, str) for x in val):
            errors.append("'evidence_phrases' must contain only strings")
        elif len(val) > 3:
            errors.append(f"'evidence_phrases' max length is 3, got {len(val)}")

    if "subtype" in obj:
        val = obj["subtype"]
        if val is not None and not isinstance(val, str):
            errors.append(f"'subtype' must be string or null, got {type(val).__name__}")
        elif isinstance(val, str):
            # Validate it's one of the 6 subtypes (v2.1 uses snake_case)
            valid_subtypes = [
                "opportunity_search",
                "customer_discovery_validation", 
                "mvp_prototyping",
                "business_model_design",
                "go_to_market_scaling",
                "funding_and_partnerships"
            ]
            # Normalize to lowercase for case-insensitive comparison
            val_lower = val.lower() if isinstance(val, str) else val
            if val_lower not in valid_subtypes:
                errors.append(
                    f"'subtype' must be one of {valid_subtypes} or null, got '{val}'"
                )
            else:
                # Normalize the subtype to lowercase in the object
                obj['subtype'] = val_lower

    # Check for unexpected extra fields (warning only)
    all_fields = set(required_fields)
    if optional_fields:
        all_fields.update(optional_fields)
    extra_fields = set(obj.keys()) - all_fields
    if extra_fields:
        errors.append(f"Unexpected extra fields: {sorted(extra_fields)}")

    return len(errors) == 0, errors


def evidence_is_substring(
    task_text: str, evidence_phrases: list[str], case_sensitive: bool = False
) -> tuple[int, int]:
    """Check if evidence phrases are literal substrings of task text.

    This enforces that LLMs must quote actual text, not paraphrase.

    Args:
        task_text: Original task text
        evidence_phrases: List of evidence phrases from LLM
        case_sensitive: Whether to do case-sensitive matching (default: False)

    Returns:
        Tuple of (num_valid, num_total)
        - num_valid: Number of evidence phrases that are substrings
        - num_total: Total number of evidence phrases
    """
    if not evidence_phrases:
        return 0, 0

    num_valid = 0
    num_total = len(evidence_phrases)

    # Normalize for comparison if case-insensitive
    if not case_sensitive:
        task_text_lower = task_text.lower()

    for phrase in evidence_phrases:
        if not phrase or not isinstance(phrase, str):
            continue

        # Check if phrase is substring
        if case_sensitive:
            if phrase in task_text:
                num_valid += 1
        else:
            if phrase.lower() in task_text_lower:
                num_valid += 1

    return num_valid, num_total


def lowercase_tags(obj: dict[str, Any]) -> dict[str, Any]:
    """Ensure tag fields are lowercase snake_case.

    LLMs sometimes return tags in different cases. This normalizes them.

    Args:
        obj: Parsed JSON object

    Returns:
        Modified object with lowercase tags
    """
    if "tags" in obj and isinstance(obj["tags"], dict):
        # Lowercase all tag values
        for key, value in obj["tags"].items():
            if isinstance(value, str):
                obj["tags"][key] = value.lower().replace(" ", "_").replace("-", "_")
            elif isinstance(value, list):
                obj["tags"][key] = [
                    v.lower().replace(" ", "_").replace("-", "_")
                    if isinstance(v, str)
                    else v
                    for v in value
                ]

    # Also normalize subtype to uppercase (should be OPP, CDV, etc.)
    if "subtype" in obj and isinstance(obj["subtype"], str):
        obj["subtype"] = obj["subtype"].upper()

    return obj


def calculate_agreement(votes: list[int]) -> tuple[float, int]:
    """Calculate agreement fraction and modal label from votes.

    Args:
        votes: List of labels (0 or 1)

    Returns:
        Tuple of (agreement_fraction, modal_label)
        - agreement_fraction: Fraction of votes for modal label (0.0 to 1.0)
        - modal_label: Most common label (0 or 1)
    """
    if not votes:
        return 0.0, 0

    # Count votes
    count_0 = votes.count(0)
    count_1 = votes.count(1)

    # Determine modal label
    modal_label = 1 if count_1 >= count_0 else 0
    modal_count = max(count_0, count_1)

    # Calculate agreement fraction
    agreement_fraction = modal_count / len(votes)

    return agreement_fraction, modal_label

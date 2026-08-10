"""Utility functions to parse the raw Apache Avro JIRA export (avro-issues.json).

The export is a JSON-lines file: every line is one issue, in the shape returned
by the Atlassian JIRA REST API. These helpers extract flat, machine-friendly
features from that nested structure so the rest of the analysis can use them
as ordinary columns. The raw JSON stays the authoritative source of the data.
"""

import json

import numpy as np
import pandas as pd

JSON_DATA_PATH = "data_sets/raw/avro-issues.json"


def load_issues(path=JSON_DATA_PATH):
    """Return the list of issues (one dict per line of the JSON-lines file)."""
    with open(path, encoding="utf-8") as json_file:
        return [json.loads(line) for line in json_file]


def issue_key(issue):
    return issue.get("key")


def issue_number_from_key(key):
    """Turn 'AVRO-1350' into the int 1350 (a creation-order / era proxy)."""
    try:
        return int(str(key).split("-")[-1])
    except (ValueError, AttributeError):
        return np.nan


def component_names(issue):
    return [c.get("name") for c in issue["fields"].get("components", [])]


def num_components(issue):
    return len(issue["fields"].get("components", []))


def has_component(issue, name):
    return 1 if name in component_names(issue) else 0


def num_affected_versions(issue):
    return len(issue["fields"].get("versions", []))


def num_fix_versions(issue):
    return len(issue["fields"].get("fixVersions", []))


def num_labels(issue):
    return len(issue["fields"].get("labels", []))


def _to_number(value, default=0.0):
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def attachment_count(issue):
    """Number of attachments (custom field; may be a string, float or None)."""
    return _to_number(issue["fields"].get("customfield_12310310"), default=0.0)


def comment_count(issue):
    return issue["fields"].get("comment", {}).get("total", 0)


def distinct_comment_authors(issue):
    return len(
        {
            comment.get("author", {}).get("name")
            for comment in issue["fields"].get("comment", {}).get("comments", [])
        }
    )


def first_response_delay_hours(issue):
    """Hours between creation and the first community response.

    The export carries the first response directly in a custom field; when that
    is missing, the timestamp of the first comment is used as a fallback.
    NOTE: this is a future event for a brand-new issue, so it is only reported
    in the exploration, not used as a model feature.
    """
    created = issue["fields"].get("created")
    if not created:
        return np.nan

    first_response = issue["fields"].get("customfield_12310220")
    if not first_response:
        comments = issue["fields"].get("comment", {}).get("comments", [])
        first_response = comments[0].get("created") if comments else None
    if not first_response:
        return np.nan

    created_dt = pd.to_datetime(created, errors="coerce", utc=True)
    response_dt = pd.to_datetime(first_response, errors="coerce", utc=True)
    if pd.isna(created_dt) or pd.isna(response_dt):
        return np.nan
    return (response_dt - created_dt).total_seconds() / 3600.0


def build_feature_frame(path=JSON_DATA_PATH):
    """Return a DataFrame (indexed by issue key) with all extracted features."""
    issues = load_issues(path)
    rows = []
    for issue in issues:
        key = issue_key(issue)
        rows.append(
            {
                "key": key,
                "issue_number": issue_number_from_key(key),
                "num_components": num_components(issue),
                "component_java": has_component(issue, "java"),
                "num_affected_versions": num_affected_versions(issue),
                "num_fix_versions": num_fix_versions(issue),
                "num_labels": num_labels(issue),
                "attachment_count": attachment_count(issue),
                "comment_count_json": comment_count(issue),
                "distinct_comment_authors": distinct_comment_authors(issue),
                "first_response_delay_hours": first_response_delay_hours(issue),
            }
        )
    frame = pd.DataFrame(rows).set_index("key")
    return frame

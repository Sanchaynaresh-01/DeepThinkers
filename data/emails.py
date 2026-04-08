"""
Email dataset for the Email Triage benchmark environment.
Loads realistic email scenarios from external JSON files.
"""

import json
import os
import random
from typing import Any

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_json(filename: str) -> list[dict[str, Any]]:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

EMAIL_DATASET = {
    "priority-classification": _load_json("easy.json"),
    "category-routing": _load_json("medium.json"),
    "full-triage-pipeline": _load_json("hard.json"),
}

def get_emails_for_task(task_name: str) -> list[dict[str, Any]]:
    """Return the email list for a given task."""
    return EMAIL_DATASET.get(task_name, [])

def get_random_email(task_name: str) -> dict[str, Any]:
    """Return a random email for the given task."""
    emails = get_emails_for_task(task_name)
    if not emails:
        raise ValueError(f"No emails found for task: {task_name}")
    return random.choice(emails)

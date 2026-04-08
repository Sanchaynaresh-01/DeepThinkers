"""
Agent graders for the Email Triage benchmark environment.

Each grader evaluates a TriageAction against ground truth and returns
a normalized score in [0.0, 1.0] with partial credit for each dimension.
"""

from typing import Any


PRIORITY_ADJACENCY = {
    "critical": {"critical": 1.0, "high": 0.5, "medium": 0.1, "low": 0.0},
    "high":     {"critical": 0.5, "high": 1.0, "medium": 0.5, "low": 0.1},
    "medium":   {"critical": 0.1, "high": 0.5, "medium": 1.0, "low": 0.5},
    "low":      {"critical": 0.0, "high": 0.1, "medium": 0.5, "low": 1.0},
}

SLA_TOLERANCE = {
    "critical": 2,   # ±2 hours
    "high":     8,   # ±8 hours
    "medium":   24,
    "low":      48,
}


def score_priority(predicted: str | None, ground_truth: str) -> float:
    """Score priority using adjacency matrix for partial credit."""
    if predicted is None:
        return 0.0
    predicted = predicted.lower().strip()
    gt = ground_truth.lower().strip()
    return PRIORITY_ADJACENCY.get(gt, {}).get(predicted, 0.0)


def score_category(predicted: str | None, ground_truth: str) -> float:
    """Score category — exact match = 1.0, no partial credit."""
    if predicted is None:
        return 0.0
    return 1.0 if predicted.lower().strip() == ground_truth.lower().strip() else 0.0


def score_routing(predicted: str | None, ground_truth: str) -> float:
    """Score routing with partial credit for partial matches."""
    if predicted is None:
        return 0.0
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()
    if p == g:
        return 1.0
    # Partial credit: check if key words match
    p_words = set(p.replace("-", " ").split())
    g_words = set(g.replace("-", " ").split())
    overlap = len(p_words & g_words)
    if overlap == 0:
        return 0.0
    return min(0.7, overlap / max(len(g_words), 1))


def score_sla(predicted: int | None, gt_priority: str, gt_sla: int) -> float:
    """Score SLA hours — full credit if within tolerance for the priority tier."""
    if predicted is None:
        return 0.0
    tolerance = SLA_TOLERANCE.get(gt_priority, 24)
    diff = abs(predicted - gt_sla)
    if diff == 0:
        return 1.0
    if diff <= tolerance:
        return max(0.0, 1.0 - (diff / (tolerance * 2)))
    return 0.0


def score_sentiment(predicted: str | None, ground_truth: str) -> float:
    """Score sentiment — exact = 1.0, adjacent = 0.5."""
    if predicted is None:
        return 0.0
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()
    if p == g:
        return 1.0
    # Adjacent sentiments
    adjacency = {
        ("urgent", "negative"): 0.5,
        ("negative", "urgent"): 0.5,
        ("positive", "neutral"): 0.5,
        ("neutral", "positive"): 0.5,
        ("neutral", "negative"): 0.3,
        ("negative", "neutral"): 0.3,
    }
    return adjacency.get((p, g), 0.0)


def score_flags(predicted: list[str], ground_truth: list[str]) -> tuple[float, float]:
    """
    Score flags: F1-like score.
    Returns (score, penalty) where penalty is for false positives on critical flags.
    """
    if not ground_truth and not predicted:
        return 1.0, 0.0

    p_set = set(f.lower() for f in predicted)
    g_set = set(f.lower() for f in ground_truth)

    if not g_set:
        # No flags expected; penalize for over-flagging
        penalty = min(1.0, len(p_set) * 0.1)
        return 1.0, penalty

    true_pos = len(p_set & g_set)
    false_neg = len(g_set - p_set)
    false_pos = len(p_set - g_set)

    precision = true_pos / max(len(p_set), 1)
    recall = true_pos / max(len(g_set), 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    # Extra penalty for missing "escalate" or "legal_risk" when they're ground truth
    critical_flags = {"escalate", "legal_risk"}
    missed_critical = critical_flags & (g_set - p_set)
    penalty = len(missed_critical) * 0.15

    return f1, min(penalty, 0.3)


def score_action_items(predicted: list[str], ground_truth: list[str]) -> float:
    """
    Score action items using keyword coverage.
    Full credit requires covering the key verbs/nouns of each expected action item.
    """
    if not ground_truth:
        return 1.0  # No action items expected — no penalty

    if not predicted:
        return 0.0

    def extract_keywords(item: str) -> set[str]:
        stop = {"the", "a", "an", "to", "and", "or", "of", "for", "in", "on", "with", "if", "needed"}
        return set(w.lower() for w in item.split() if w.lower() not in stop and len(w) > 2)

    covered = 0
    for gt_item in ground_truth:
        gt_kw = extract_keywords(gt_item)
        for pred_item in predicted:
            pred_kw = extract_keywords(pred_item)
            if gt_kw and len(gt_kw & pred_kw) / len(gt_kw) >= 0.5:
                covered += 1
                break

    return covered / len(ground_truth)


# ─────────────────────────────────────────────
# Task-specific graders
# ─────────────────────────────────────────────

def grade_priority_classification(action: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, float]:
    """
    Easy task: only priority is graded. Max score = 1.0.
    """
    p_score = score_priority(action.get("priority"), ground_truth["priority"])
    # Bonus: correct sentiment adds small signal
    s_score = score_sentiment(action.get("sentiment"), ground_truth.get("sentiment", "neutral"))

    total = p_score * 0.85 + s_score * 0.15
    return {
        "total": round(min(total, 1.0), 4),
        "priority_score": round(p_score, 4),
        "sentiment_score": round(s_score, 4),
        "category_score": 0.0,
        "routing_score": 0.0,
        "action_items_score": 0.0,
        "sla_score": 0.0,
        "flags_score": 0.0,
        "penalty": 0.0,
    }


def grade_category_routing(action: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, float]:
    """
    Medium task: priority + category + routing + action_items. Max score = 1.0.
    """
    p_score = score_priority(action.get("priority"), ground_truth["priority"])
    c_score = score_category(action.get("category"), ground_truth["category"])
    r_score = score_routing(action.get("route_to"), ground_truth["route_to"])
    ai_score = score_action_items(
        action.get("action_items", []),
        ground_truth.get("action_items", [])
    )
    f_score, f_penalty = score_flags(
        action.get("flags", []),
        ground_truth.get("flags", [])
    )

    # Weighted combination
    total = (
        p_score  * 0.25 +
        c_score  * 0.30 +
        r_score  * 0.20 +
        ai_score * 0.15 +
        f_score  * 0.10
    ) - f_penalty

    return {
        "total": round(max(0.0, min(total, 1.0)), 4),
        "priority_score": round(p_score, 4),
        "category_score": round(c_score, 4),
        "routing_score": round(r_score, 4),
        "action_items_score": round(ai_score, 4),
        "flags_score": round(f_score, 4),
        "sla_score": 0.0,
        "sentiment_score": 0.0,
        "penalty": round(f_penalty, 4),
    }


def grade_full_triage_pipeline(action: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, float]:
    """
    Hard task: all fields graded. Max score = 1.0.
    """
    p_score  = score_priority(action.get("priority"), ground_truth["priority"])
    c_score  = score_category(action.get("category"), ground_truth["category"])
    r_score  = score_routing(action.get("route_to"), ground_truth["route_to"])
    ai_score = score_action_items(
        action.get("action_items", []),
        ground_truth.get("action_items", [])
    )
    sla_score = score_sla(
        action.get("sla_hours"),
        ground_truth["priority"],
        ground_truth.get("sla_hours", 24)
    )
    s_score = score_sentiment(action.get("sentiment"), ground_truth.get("sentiment", "neutral"))
    f_score, f_penalty = score_flags(
        action.get("flags", []),
        ground_truth.get("flags", [])
    )

    # Weighted combination — all fields matter
    total = (
        p_score   * 0.20 +
        c_score   * 0.20 +
        r_score   * 0.15 +
        ai_score  * 0.20 +
        sla_score * 0.10 +
        s_score   * 0.05 +
        f_score   * 0.10
    ) - f_penalty

    # Extra penalty for missing critical priority on critical emails
    if ground_truth["priority"] == "critical" and p_score < 0.5:
        total -= 0.15

    return {
        "total": round(max(0.0, min(total, 1.0)), 4),
        "priority_score": round(p_score, 4),
        "category_score": round(c_score, 4),
        "routing_score": round(r_score, 4),
        "action_items_score": round(ai_score, 4),
        "sla_score": round(sla_score, 4),
        "sentiment_score": round(s_score, 4),
        "flags_score": round(f_score, 4),
        "penalty": round(f_penalty, 4),
    }


TASK_GRADERS = {
    "priority-classification": grade_priority_classification,
    "category-routing": grade_category_routing,
    "full-triage-pipeline": grade_full_triage_pipeline,
}

TASK_MAX_STEPS = {
    "priority-classification": 10,
    "category-routing": 15,
    "full-triage-pipeline": 20,
}


def grade(task_name: str, action: dict[str, Any], ground_truth: dict[str, Any]) -> dict[str, float]:
    """Dispatch to the correct grader and return scores."""
    grader = TASK_GRADERS.get(task_name)
    if grader is None:
        raise ValueError(f"Unknown task: {task_name}")
    return grader(action, ground_truth)

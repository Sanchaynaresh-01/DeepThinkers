"""
Offline test: simulate what a smart LLM would likely predict for each easy email,
then grade it against ground truth. This verifies the benchmark produces varied,
realistic scores WITHOUT needing the server or HF API.
"""
import sys, os, io, json

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.emails import get_emails_for_task
from server.graders import grade

# What a smart 72B model would LIKELY predict based on reading each email
model_predictions = {
    "pc-001": {"priority": "low",      "sentiment": "neutral"},   # Model sees "informational, no immediate action" -> low
    "pc-002": {"priority": "medium",   "sentiment": "neutral"},   # Model sees "not looking to leave, 6 weeks" -> medium
    "pc-003": {"priority": "low",      "sentiment": "neutral"},   # Model sees "friendly reminder" -> low  
    "pc-004": {"priority": "high",     "sentiment": "positive"},  # Model sees "$180K deal blocker" -> high
    "pc-005": {"priority": "critical", "sentiment": "urgent"},    # Model sees "unusual login, security" -> critical
    "pc-006": {"priority": "low",      "sentiment": "positive"},  # Model sees "no rush, love the product" -> low (MATCH!)
    "pc-007": {"priority": "medium",   "sentiment": "negative"},  # Model sees "price increase" -> medium
    "pc-008": {"priority": "low",      "sentiment": "neutral"},   # Model sees "resolved, thanks" -> low
}

print("=" * 70)
print("OFFLINE GRADER TEST: Easy Task (priority-classification)")
print("=" * 70)

emails = get_emails_for_task("priority-classification")
total_score = 0

for e in emails:
    eid = e["email_id"]
    pred = model_predictions.get(eid, {"priority": "medium", "sentiment": "neutral"})
    gt = e["ground_truth"]
    scores = grade("priority-classification", pred, gt)
    total_score += scores["total"]
    
    match_p = "MATCH" if pred["priority"] == gt["priority"] else "MISS"
    match_s = "MATCH" if pred["sentiment"] == gt.get("sentiment") else "MISS"
    
    print(f"\n[{eid}] Subject: {e['subject'][:50]}...")
    print(f"  Model:  priority={pred['priority']:<10} sentiment={pred['sentiment']:<10}")
    print(f"  Truth:  priority={gt['priority']:<10} sentiment={gt.get('sentiment','?'):<10}")
    print(f"  Result: priority={match_p} sentiment={match_s} -> score={scores['total']:.3f}")

avg = total_score / len(emails)
print(f"\n{'=' * 70}")
print(f"Average Score: {avg:.3f}")
print(f"Score Range: Healthy={'YES' if 0.35 <= avg <= 0.85 else 'NO'} (target: 0.35-0.85)")
print(f"{'=' * 70}")

# Also verify medium and hard datasets still load correctly
for task in ["category-routing", "full-triage-pipeline"]:
    emails = get_emails_for_task(task)
    print(f"\n[CHECK] {task}: {len(emails)} emails loaded OK")
    for e in emails:
        assert "ground_truth" in e, f"Missing ground_truth in {e['email_id']}"
    print(f"  All ground_truth keys present")

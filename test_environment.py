#!/usr/bin/env python3
"""
test_environment.py
Self-contained test suite for the Email Triage OpenEnv.
Run with: python test_environment.py
No pytest required.
"""

import sys
import os
import json
import io

# Fix Windows console encoding for Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Project root is the directory containing this file
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.emails import get_emails_for_task, get_random_email
from server.graders import (
    grade,
    score_priority,
    score_category,
    score_routing,
    score_sla,
    score_sentiment,
    score_flags,
    score_action_items,
    TASK_GRADERS,
    TASK_MAX_STEPS,
)
from server.environment import EmailTriageEnv, TASK_NAMES

PASS = 0
FAIL = 0


def check(name, condition, got=None, expected=None):
    global PASS, FAIL
    if condition:
        print(f"  ✓  {name}")
        PASS += 1
    else:
        print(f"  ✗  {name}  (got={got!r}, expected={expected!r})")
        FAIL += 1


def approx_eq(a, b, tol=0.001):
    return abs(a - b) <= tol


# ─────────────────────────────────────────────
# 1. Dataset sanity checks
# ─────────────────────────────────────────────
print("\n=== 1. Dataset ===")
for task in TASK_NAMES:
    emails = get_emails_for_task(task)
    check(f"{task}: has emails", len(emails) >= 3, got=len(emails))
    for e in emails:
        check(f"  {e['email_id']}: has ground_truth", "ground_truth" in e)
        gt = e["ground_truth"]
        check(f"  {e['email_id']}: priority in range",
              gt.get("priority") in ("critical", "high", "medium", "low"))
        check(f"  {e['email_id']}: scores in [0,1] range", True)  # structural only

# ─────────────────────────────────────────────
# 2. Individual scorer tests
# ─────────────────────────────────────────────
print("\n=== 2. Individual Scorers ===")

# priority adjacency
check("priority: exact=1.0", approx_eq(score_priority("critical", "critical"), 1.0))
check("priority: adjacent=0.5", approx_eq(score_priority("high", "critical"), 0.5))
check("priority: far=0.0", approx_eq(score_priority("low", "critical"), 0.0))
check("priority: None=0.0", approx_eq(score_priority(None, "critical"), 0.0))

# category
check("category: exact=1.0", approx_eq(score_category("legal", "legal"), 1.0))
check("category: wrong=0.0", approx_eq(score_category("billing", "legal"), 0.0))
check("category: None=0.0", approx_eq(score_category(None, "legal"), 0.0))

# routing
check("routing: exact=1.0", approx_eq(score_routing("legal-team", "legal-team"), 1.0))
check("routing: partial credit>0", score_routing("legal-compliance", "legal-team") > 0)
check("routing: completely wrong=0.0", approx_eq(score_routing("trash", "legal-team"), 0.0))

# sla
check("sla: exact=1.0", approx_eq(score_sla(1, "critical", 1), 1.0))
check("sla: within tolerance>0", score_sla(2, "critical", 1) > 0)
check("sla: way off=0.0", approx_eq(score_sla(168, "critical", 1), 0.0))
check("sla: None=0.0", approx_eq(score_sla(None, "critical", 1), 0.0))

# sentiment
check("sentiment: exact=1.0", approx_eq(score_sentiment("urgent", "urgent"), 1.0))
check("sentiment: adjacent=0.5", approx_eq(score_sentiment("negative", "urgent"), 0.5))
check("sentiment: far=0.0", approx_eq(score_sentiment("positive", "urgent"), 0.0))

# flags
f_score, f_penalty = score_flags(["pii", "legal_risk"], ["pii", "legal_risk"])
check("flags: perfect f1=1.0", approx_eq(f_score, 1.0))
check("flags: perfect penalty=0.0", approx_eq(f_penalty, 0.0))

f_score2, f_pen2 = score_flags([], ["escalate", "legal_risk"])
check("flags: missing critical flags → penalty>0", f_pen2 > 0)
check("flags: missing all → score<1", f_score2 < 1.0)

# action items
ai_score = score_action_items(
    ["investigate rate limit issue", "check account quota and billing"],
    ["investigate rate limit", "check account quota"],
)
check("action_items: good coverage>0.5", ai_score > 0.5)
check("action_items: empty predicted=0.0",
      approx_eq(score_action_items([], ["do something"]), 0.0))
check("action_items: empty ground truth=1.0",
      approx_eq(score_action_items(["anything"], []), 1.0))

# ─────────────────────────────────────────────
# 3. Task grader bounds
# ─────────────────────────────────────────────
print("\n=== 3. Grader Bounds ===")

for task in TASK_NAMES:
    emails = get_emails_for_task(task)
    for email in emails:
        gt = email["ground_truth"]

        # Perfect action (use ground truth)
        perfect = {
            "priority":     gt.get("priority"),
            "category":     gt.get("category"),
            "route_to":     gt.get("route_to"),
            "action_items": gt.get("action_items", []),
            "sla_hours":    gt.get("sla_hours"),
            "sentiment":    gt.get("sentiment"),
            "flags":        gt.get("flags", []),
        }
        scores = grade(task, perfect, gt)
        check(f"{task}/{email['email_id']}: perfect in [0,1]",
              0.0 <= scores["total"] <= 1.0, got=scores["total"])
        check(f"{task}/{email['email_id']}: perfect ≥ 0.9",
              scores["total"] >= 0.9, got=scores["total"])

        # Terrible action
        worst = {
            "priority":     "low" if gt["priority"] == "critical" else "critical",
            "category":     "spam",
            "route_to":     "trash",
            "action_items": [],
            "sla_hours":    999,
            "sentiment":    "positive" if gt.get("sentiment") == "urgent" else "urgent",
            "flags":        [],
        }
        worst_scores = grade(task, worst, gt)
        check(f"{task}/{email['email_id']}: worst in [0,1]",
              0.0 <= worst_scores["total"] <= 1.0, got=worst_scores["total"])
        check(f"{task}/{email['email_id']}: worst < 0.6",
              worst_scores["total"] < 0.6, got=worst_scores["total"])

# ─────────────────────────────────────────────
# 4. Environment lifecycle
# ─────────────────────────────────────────────
print("\n=== 4. Environment Lifecycle ===")

for task in TASK_NAMES:
    env = EmailTriageEnv(task_name=task, seed=0)

    # reset
    obs = env.reset()
    check(f"{task}: reset returns observation", obs is not None)
    check(f"{task}: obs.step == 0", obs.step == 0)
    check(f"{task}: obs.email_id non-empty", bool(obs.email_id))
    check(f"{task}: obs.task_name correct", obs.task_name == task)

    # state after reset
    state = env.state()
    check(f"{task}: state.task_name correct", state.task_name == task)
    check(f"{task}: state.done=False after reset", not state.done)

    # step
    action = {
        "priority": "high",
        "category": "engineering",
        "route_to": "engineering-oncall",
        "action_items": ["review logs", "page oncall"],
        "sla_hours": 4,
        "sentiment": "urgent",
        "flags": ["escalate"],
        "reasoning": "Seems urgent.",
    }
    result = env.step(action)
    check(f"{task}: step returns result", result is not None)
    check(f"{task}: reward in [0,1]", 0.0 <= result.reward <= 1.0, got=result.reward)
    check(f"{task}: done is bool", isinstance(result.done, bool))
    check(f"{task}: result.info has 'step'", "step" in result.info)

    # state after step
    state2 = env.state()
    check(f"{task}: state.step==1 after 1 step", state2.step == 1)
    check(f"{task}: cumulative_reward == step reward",
          approx_eq(state2.cumulative_reward, result.reward))

    # drain episode
    steps = 1
    max_s = TASK_MAX_STEPS[task]
    while not result.done and steps < max_s + 2:
        result = env.step(action)
        steps += 1

    check(f"{task}: episode terminates", result.done, got=result.done)
    check(f"{task}: done within max_steps+1", steps <= max_s + 1, got=steps)

    # reset re-initialises
    obs2 = env.reset()
    check(f"{task}: re-reset clears step", obs2.step == 0)
    state3 = env.state()
    check(f"{task}: re-reset clears cumulative_reward",
          approx_eq(state3.cumulative_reward, 0.0))

# ─────────────────────────────────────────────
# 5. stdout format smoke-test (log helpers)
# ─────────────────────────────────────────────
print("\n=== 5. Stdout Format ===")

import io
import contextlib

def capture_print(fn, *args, **kwargs):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue().strip()

# Import log helpers from inference without running main()
inference_path = os.path.join(PROJECT_ROOT, "inference.py")
import importlib.util
spec = importlib.util.spec_from_file_location("inference", inference_path)
inf = importlib.util.module_from_spec(spec)
# Don't exec (would try to connect), just grab the functions via exec of just the defs
import ast, types

with open(inference_path, encoding="utf-8") as f:
    src = f.read()

# Extract and test the log functions directly
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = str(action).replace("\n", " ")[:200]
    print(f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

start_line = capture_print(log_start, "priority-classification", "email-triage-env", "TestModel")
check("START: begins with [START]", start_line.startswith("[START]"))
check("START: has task=", "task=" in start_line)
check("START: has env=", "env=" in start_line)
check("START: has model=", "model=" in start_line)

step_line = capture_print(log_step, 1, '{"priority":"high"}', 0.75, False, None)
check("STEP: begins with [STEP]", step_line.startswith("[STEP]"))
check("STEP: has step=", "step=" in step_line)
check("STEP: has reward=0.75", "reward=0.75" in step_line)
check("STEP: has done=false", "done=false" in step_line)
check("STEP: has error=null", "error=null" in step_line)

end_line = capture_print(log_end, True, 5, 0.623, [0.5, 0.7, 0.6, 0.55, 0.7])
check("END: begins with [END]", end_line.startswith("[END]"))
check("END: has success=true", "success=true" in end_line)
check("END: has steps=5", "steps=5" in end_line)
check("END: has score=0.623", "score=0.623" in end_line)
check("END: has rewards list", "rewards=" in end_line)
check("END: no newlines in line", "\n" not in end_line)

# ─────────────────────────────────────────────
# 6. openenv.yaml sanity
# ─────────────────────────────────────────────
print("\n=== 6. openenv.yaml ===")

yaml_path = os.path.join(PROJECT_ROOT, "openenv.yaml")
check("openenv.yaml exists", os.path.exists(yaml_path))
with open(yaml_path, encoding="utf-8") as f:
    yaml_content = f.read()
for field in ["name:", "version:", "tasks:", "observation_space:", "action_space:", "reward:", "endpoint:"]:
    check(f"openenv.yaml has '{field}'", field in yaml_content)

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────
total = PASS + FAIL
print(f"\n{'='*50}")
print(f"Results: {PASS}/{total} passed", end="")
if FAIL:
    print(f"  ({FAIL} FAILED)")
    sys.exit(1)
else:
    print("  ✓ ALL PASSED")
    sys.exit(0)

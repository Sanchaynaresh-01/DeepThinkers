#!/usr/bin/env python3
"""
validate.py — Pre-submission validation script for OpenEnv.
Checks all items from the pre-submission checklist:
  1. openenv.yaml exists and is well-formed
  2. Typed Pydantic models exist
  3. step() / reset() / state() work correctly
  4. 3+ tasks with graders, scores in [0.0, 1.0]
  5. Dockerfile exists and is parseable
  6. inference.py exists at root
  7. Environment variables documented
  8. Stdout log format compliance
"""

import json
import os
import sys
import time
import subprocess
import importlib

# ── Utilities ────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
WARN = 0


def check(name: str, condition: bool, got=None, expected=None, critical: bool = True):
    global PASS, FAIL, WARN
    if condition:
        print(f"  ✓  {name}")
        PASS += 1
    elif critical:
        msg = f"  ✗  {name}"
        if got is not None:
            msg += f"  (got={got!r}"
            if expected is not None:
                msg += f", expected={expected!r}"
            msg += ")"
        print(msg)
        FAIL += 1
    else:
        print(f"  ⚠  {name} (warning)")
        WARN += 1


ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    global PASS, FAIL, WARN

    # ─────────────────────────────────────────────
    # 1. File structure
    # ─────────────────────────────────────────────
    print("\n=== 1. File Structure ===")

    required_files = [
        "openenv.yaml",
        "Dockerfile",
        "README.md",
        "requirements.txt",
        "inference.py",
        "server/app.py",
        "server/environment.py",
        "server/models.py",
        "server/graders.py",
        "server/__init__.py",
        "data/emails.py",
        "data/__init__.py",
    ]

    for f in required_files:
        path = os.path.join(ROOT, f)
        check(f"File exists: {f}", os.path.exists(path))

    # ─────────────────────────────────────────────
    # 2. openenv.yaml validation
    # ─────────────────────────────────────────────
    print("\n=== 2. openenv.yaml ===")

    yaml_path = os.path.join(ROOT, "openenv.yaml")
    try:
        # Use simple YAML parsing (no PyYAML dependency needed)
        with open(yaml_path, encoding="utf-8") as f:
            yaml_content = f.read()
        check("openenv.yaml is readable", True)

        required_fields = ["name:", "version:", "tasks:", "observation_space:", "action_space:", "reward:", "endpoint:"]
        for field in required_fields:
            check(f"openenv.yaml has '{field}'", field in yaml_content)

        # Check for openenv tag
        check("openenv.yaml has 'openenv' tag", "openenv" in yaml_content)

        # Count tasks
        task_count = yaml_content.count("- name:")
        check(f"openenv.yaml defines 3+ tasks", task_count >= 3, got=task_count, expected=">=3")

    except FileNotFoundError:
        check("openenv.yaml exists", False)

    # ─────────────────────────────────────────────
    # 3. Pydantic typed models
    # ─────────────────────────────────────────────
    print("\n=== 3. Typed Models ===")

    sys.path.insert(0, ROOT)

    try:
        from server.models import (
            EmailObservation, TriageAction, TriageReward,
            StepResult, EpisodeState,
            Priority, Category, Sentiment,
        )
        check("Import EmailObservation", True)
        check("Import TriageAction", True)
        check("Import TriageReward", True)
        check("Import StepResult", True)
        check("Import EpisodeState", True)
        check("Import Priority enum", True)
        check("Import Category enum", True)
        check("Import Sentiment enum", True)

        # Verify they're proper Pydantic models
        obs = EmailObservation(email_id="test", subject="Test", sender="a@b.com")
        check("EmailObservation instantiable", obs is not None)
        check("EmailObservation has model_dump", hasattr(obs, "model_dump"))

        action = TriageAction(priority="critical", category="engineering")
        check("TriageAction instantiable", action is not None)
        check("TriageAction has model_dump", hasattr(action, "model_dump"))

        reward = TriageReward(total=0.75)
        check("TriageReward instantiable", reward is not None)

    except Exception as e:
        check(f"Import models: {e}", False)

    # ─────────────────────────────────────────────
    # 4. Environment: reset() / step() / state()
    # ─────────────────────────────────────────────
    print("\n=== 4. Environment Interface ===")

    try:
        from server.environment import EmailTriageEnv, TASK_NAMES
        from server.graders import TASK_MAX_STEPS

        check("3+ task names defined", len(TASK_NAMES) >= 3, got=len(TASK_NAMES))

        for task in TASK_NAMES:
            env = EmailTriageEnv(task_name=task, seed=42)

            # reset()
            obs = env.reset()
            check(f"{task}: reset() returns observation", obs is not None)
            check(f"{task}: obs has email_id", hasattr(obs, "email_id") and obs.email_id)
            check(f"{task}: obs.step == 0", obs.step == 0)

            # state()
            state = env.state()
            check(f"{task}: state() returns EpisodeState", state is not None)
            check(f"{task}: state.done == False after reset", not state.done)
            check(f"{task}: state.task_name correct", state.task_name == task)

            # step()
            action = {
                "priority": "high",
                "category": "engineering",
                "route_to": "engineering-oncall",
                "action_items": ["review logs"],
                "sla_hours": 4,
                "sentiment": "urgent",
                "flags": ["escalate"],
                "reasoning": "Test action.",
            }
            result = env.step(action)
            check(f"{task}: step() returns result", result is not None)
            check(f"{task}: reward in [0,1]",
                  0.0 <= result.reward <= 1.0, got=result.reward)
            check(f"{task}: done is bool", isinstance(result.done, bool))
            check(f"{task}: result has observation", result.observation is not None)

            # state after step
            state2 = env.state()
            check(f"{task}: state.step==1 after step", state2.step == 1)

    except Exception as e:
        check(f"Environment interface: {e}", False)

    # ─────────────────────────────────────────────
    # 5. Graders: 3+ tasks, scores in [0.0, 1.0]
    # ─────────────────────────────────────────────
    print("\n=== 5. Task Graders ===")

    try:
        from server.graders import grade, TASK_GRADERS
        from data.emails import get_emails_for_task

        check("3+ graders defined", len(TASK_GRADERS) >= 3, got=len(TASK_GRADERS))

        for task in TASK_NAMES:
            emails = get_emails_for_task(task)
            check(f"{task}: has emails for grading", len(emails) >= 3, got=len(emails))

            for email in emails:
                gt = email["ground_truth"]

                # Perfect action
                perfect = {
                    "priority": gt.get("priority"),
                    "category": gt.get("category"),
                    "route_to": gt.get("route_to"),
                    "action_items": gt.get("action_items", []),
                    "sla_hours": gt.get("sla_hours"),
                    "sentiment": gt.get("sentiment"),
                    "flags": gt.get("flags", []),
                }
                scores = grade(task, perfect, gt)
                check(f"{task}/{email['email_id']}: perfect score in [0,1]",
                      0.0 <= scores["total"] <= 1.0, got=scores["total"])
                check(f"{task}/{email['email_id']}: perfect score >= 0.9",
                      scores["total"] >= 0.9, got=scores["total"])

                # Worst action
                worst = {
                    "priority": "low" if gt["priority"] == "critical" else "critical",
                    "category": "spam",
                    "route_to": "trash",
                    "action_items": [],
                    "sla_hours": 999,
                    "sentiment": "positive" if gt.get("sentiment") == "urgent" else "urgent",
                    "flags": [],
                }
                worst_scores = grade(task, worst, gt)
                check(f"{task}/{email['email_id']}: worst score in [0,1]",
                      0.0 <= worst_scores["total"] <= 1.0, got=worst_scores["total"])

    except Exception as e:
        check(f"Grader validation: {e}", False)

    # ─────────────────────────────────────────────
    # 6. Dockerfile validation
    # ─────────────────────────────────────────────
    print("\n=== 6. Dockerfile ===")

    dockerfile_path = os.path.join(ROOT, "Dockerfile")
    try:
        with open(dockerfile_path, encoding="utf-8") as f:
            dockerfile = f.read()
        check("Dockerfile is readable", True)
        check("Dockerfile has FROM", "FROM" in dockerfile)
        check("Dockerfile has EXPOSE 7860", "EXPOSE 7860" in dockerfile)
        check("Dockerfile has CMD", "CMD" in dockerfile)
        check("Dockerfile copies requirements", "requirements.txt" in dockerfile)
        check("Dockerfile has HEALTHCHECK", "HEALTHCHECK" in dockerfile)
    except FileNotFoundError:
        check("Dockerfile exists", False)

    # ─────────────────────────────────────────────
    # 7. inference.py validation
    # ─────────────────────────────────────────────
    print("\n=== 7. inference.py ===")

    inference_path = os.path.join(ROOT, "inference.py")
    try:
        with open(inference_path, encoding="utf-8") as f:
            inf_content = f.read()
        check("inference.py is readable", True)
        check("inference.py uses OpenAI client", "OpenAI" in inf_content or "openai" in inf_content)
        check("inference.py reads HF_TOKEN", "HF_TOKEN" in inf_content)
        check("inference.py reads API_BASE_URL", "API_BASE_URL" in inf_content)
        check("inference.py reads MODEL_NAME", "MODEL_NAME" in inf_content)
        check("inference.py has [START] log", "[START]" in inf_content)
        check("inference.py has [STEP] log", "[STEP]" in inf_content)
        check("inference.py has [END] log", "[END]" in inf_content)
        check("inference.py has main()", "def main(" in inf_content)
    except FileNotFoundError:
        check("inference.py exists at root", False)

    # ─────────────────────────────────────────────
    # 8. Stdout format compliance
    # ─────────────────────────────────────────────
    print("\n=== 8. Stdout Log Format ===")

    import io
    import contextlib

    def capture_print(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
        return buf.getvalue().strip()

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
    # 9. README checks
    # ─────────────────────────────────────────────
    print("\n=== 9. README.md ===")

    readme_path = os.path.join(ROOT, "README.md")
    try:
        with open(readme_path, encoding="utf-8") as f:
            readme = f.read()
        check("README.md is readable", True)
        check("README has environment description", "environment" in readme.lower() or "description" in readme.lower())
        check("README has action space", "action" in readme.lower() and "space" in readme.lower())
        check("README has observation space", "observation" in readme.lower() and "space" in readme.lower())
        check("README has task descriptions", "task" in readme.lower())
        check("README has setup instructions", "setup" in readme.lower() or "usage" in readme.lower())
        check("README has baseline scores", "baseline" in readme.lower() and "score" in readme.lower())

        # HF Space frontmatter
        check("README has HF Space frontmatter", readme.startswith("---"))
        check("README has sdk: docker", "sdk: docker" in readme)
        check("README has app_port", "app_port:" in readme)
        check("README has openenv tag in frontmatter", "openenv" in readme.split("---")[1] if readme.startswith("---") and readme.count("---") >= 2 else False)
    except FileNotFoundError:
        check("README.md exists", False)

    # ─────────────────────────────────────────────
    # 10. Environment variable config
    # ─────────────────────────────────────────────
    print("\n=== 10. Environment Variables ===")

    check("API_BASE_URL documented", "API_BASE_URL" in readme if 'readme' in dir() else False)
    check("MODEL_NAME documented", "MODEL_NAME" in readme if 'readme' in dir() else False)
    check("HF_TOKEN documented", "HF_TOKEN" in readme if 'readme' in dir() else False)

    # ─────────────────────────────────────────────
    # Summary
    # ─────────────────────────────────────────────
    total = PASS + FAIL + WARN
    print(f"\n{'='*60}")
    print(f"VALIDATION RESULTS: {PASS}/{total} passed", end="")
    if WARN:
        print(f"  ({WARN} warnings)", end="")
    if FAIL:
        print(f"  ({FAIL} FAILED)")
        print(f"\n❌ VALIDATION FAILED — fix {FAIL} issue(s) before submitting.")
        sys.exit(1)
    else:
        print(f"\n\n✅ ALL CHECKS PASSED — ready to submit!")
        sys.exit(0)


if __name__ == "__main__":
    main()

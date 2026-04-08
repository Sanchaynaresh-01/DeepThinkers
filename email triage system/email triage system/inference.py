"""
Inference Script — Email Triage OpenEnv
=======================================
"""

import io
import json
import os
import sys
import textwrap
import time
from typing import Any, Optional

# Auto-load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Global results storage for metrics
results = []

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import requests
from openai import OpenAI

# ── Configuration ────────────────────────────────────────────────────────────
API_KEY       = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"
API_BASE_URL  = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME    = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL  = os.getenv("ENV_BASE_URL", "http://localhost:7860")
BENCHMARK     = "email-triage-env"
TEMPERATURE   = 0.3
MAX_TOKENS    = 800

TASKS = [
    "priority-classification",
    "category-routing",
    "full-triage-pipeline",
]

TASK_MAX_STEPS = {
    "priority-classification": 10,
    "category-routing": 15,
    "full-triage-pipeline": 20,
}

SUCCESS_THRESHOLD = 0.4

# ── Logging helpers ──────────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    action_clean = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── Environment HTTP client ──────────────────────────────────────────────────

def env_reset(task: str, session_id: str = "default") -> dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task": task, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict[str, Any], session_id: str = "default") -> dict[str, Any]:
    resp = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action, "session_id": session_id},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()

# ── Prompts ──────────────────────────────────────────────────────────────────

def build_system_prompt(task: str) -> str:
    base = textwrap.dedent("""
        You are an expert email triage agent. You receive email details and must make
        structured triage decisions. Always respond with a valid JSON object only —
        no markdown, no explanation, just the JSON.
    """).strip()

    task_instructions = {
        "priority-classification": textwrap.dedent("""
            Task: Classify the email's PRIORITY and SENTIMENT.

            Required JSON fields:
            - "priority": one of "critical", "high", "medium", "low"
            - "sentiment": one of "positive", "neutral", "negative", "urgent"
            - "reasoning": brief explanation (1-2 sentences)

            Priority guidelines:
            - critical: production outages, security breaches, legal deadlines <2h, ransomware
            - high: legal threats, important clients, billing disputes, time-sensitive sales
            - medium: routine support, billing questions, feature requests
            - low: newsletters, internal social, general inquiries
        """),

        "category-routing": textwrap.dedent("""
            Task: Classify PRIORITY, CATEGORY, ROUTING, ACTION ITEMS, and FLAGS.

            Required JSON fields:
            - "priority": one of "critical", "high", "medium", "low"
            - "category": one of "support", "billing", "sales", "legal", "hr", "engineering", "spam", "other"
            - "route_to": target team queue (e.g., "technical-support", "billing-disputes", "legal-team", "engineering-oncall")
            - "action_items": list of specific actions to take (3-5 items)
            - "flags": list from ["pii", "legal_risk", "financial", "escalate"] (can be empty)
            - "reasoning": brief explanation
        """),

        "full-triage-pipeline": textwrap.dedent("""
            Task: Complete triage pipeline — all fields required.

            Required JSON fields:
            - "priority": one of "critical", "high", "medium", "low"
            - "category": one of "support", "billing", "sales", "legal", "hr", "engineering", "spam", "other"
            - "route_to": specific team queue
            - "action_items": comprehensive list of concrete actions (4-6 items)
            - "sla_hours": integer hours for SLA (critical=1-2, high=4-8, medium=24, low=48-168)
            - "sentiment": one of "positive", "neutral", "negative", "urgent"
            - "flags": relevant from ["pii", "legal_risk", "financial", "escalate"]
            - "reasoning": detailed explanation (2-3 sentences)
        """),
    }

    return base + "\n\n" + task_instructions.get(task, "")


def build_user_prompt(obs: dict[str, Any], step: int) -> str:
    feedback = obs.get("feedback", "")
    prev_actions = obs.get("previous_actions", [])
    prev_block = ""
    if prev_actions:
        prev_block = "\nPrevious actions this episode:\n"
        for pa in prev_actions[-2:]:
            prev_block += f"  Step {pa.get('step')}: reward={pa.get('reward', 0):.2f}\n"

    feedback_block = f"\nFeedback from last step: {feedback}" if feedback else ""

    return textwrap.dedent(f"""
        Step: {step}
        Email ID: {obs.get('email_id')}
        Timestamp: {obs.get('timestamp')}
        Thread length: {obs.get('thread_length')} email(s)
        Has attachments: {obs.get('has_attachments')}
        Sender: {obs.get('sender')} [{obs.get('sender_domain')}]
        Subject: {obs.get('subject')}

        Body:
        {obs.get('body')}
        {prev_block}{feedback_block}

        Respond with your triage decision as a JSON object.
    """).strip()

# ── LLM call ─────────────────────────────────────────────────────────────────

def get_triage_action(client: OpenAI, task: str, obs: dict[str, Any], step: int):

    system_prompt = build_system_prompt(task)
    user_prompt = build_user_prompt(obs, step)

    try:
        # ✅ ADD latency tracking
        start_time = time.time()

        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )

        latency = time.time() - start_time

        text = (completion.choices[0].message.content or "").strip()

        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

        parsed = json.loads(text)
        return parsed, latency, True

    except json.JSONDecodeError:
        return {
            "priority": "medium",
            "category": "support",
            "route_to": "customer-support",
            "action_items": ["review email"],
            "sla_hours": 24,
            "sentiment": "neutral",
            "flags": [],
            "reasoning": "fallback",
        }, 0.0, False

    except Exception as exc:
        print(f"[ERROR] LLM API call failed: {type(exc).__name__}: {exc}", flush=True)
        return {
            "priority": "medium",
            "category": "other",
            "route_to": "general-queue",
            "action_items": ["manual check"],
            "sla_hours": 48,
            "sentiment": "neutral",
            "flags": [],
            "reasoning": "error fallback",
        }, 0.0, False

# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(client: OpenAI, task: str):

    session_id = f"{task}-{int(time.time())}"
    max_steps = TASK_MAX_STEPS[task]

    log_start(task, BENCHMARK, MODEL_NAME)

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    done = False

    reset_resp = env_reset(task, session_id)
    obs = reset_resp["observation"]

    for step in range(1, max_steps + 1):
        if done:
            break

        # ✅ UPDATED call
        action, latency, valid_json = get_triage_action(client, task, obs, step)

        action_str = json.dumps(action, separators=(",", ":"))

        try:
            step_resp = env_step(action, session_id)
            reward = step_resp.get("reward", 0.0)
            done   = step_resp.get("done", False)
            obs    = step_resp.get("observation", obs)
        except Exception as e:
            reward = 0.0
            done = True

        rewards.append(reward)
        steps_taken = step

        # ✅ ADD metrics logging
        results.append({
            "task": task,
            "step": step,
            "reward": reward,
            "latency": latency,
            "valid_json": valid_json,
            "email_id": obs.get("email_id"),
        })

        log_step(step, action_str, reward, done, None)

    if rewards:
        score = sum(rewards) / len(rewards)

    success = score >= SUCCESS_THRESHOLD

    log_end(success, steps_taken, score, rewards)
    return success, steps_taken, score, rewards

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    try:
        requests.get(f"{ENV_BASE_URL}/health", timeout=5)
    except requests.exceptions.ConnectionError:
        print(f"\n[CRITICAL ERROR] The environment server at {ENV_BASE_URL} is NOT running!")
        print("Please start the server first by running: .\\run_local.bat\n")
        sys.exit(1)

    if API_KEY == "dummy":
        print("\n[WARNING] No HF_TOKEN or API_KEY found!")
        print("Create a .env file with: HF_TOKEN=hf_your_token_here")
        print("Or set the environment variable: set HF_TOKEN=hf_your_token_here\n")
        sys.exit(1)

    print(f"[CONFIG] Model: {MODEL_NAME}", flush=True)
    print(f"[CONFIG] API: {API_BASE_URL}", flush=True)
    print(f"[CONFIG] Token: {API_KEY[:8]}...{API_KEY[-4:]}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    overall_results = []

    for task in TASKS:
        success, steps, score, rewards = run_episode(client, task)

        overall_results.append({
            "task": task,
            "success": success,
            "score": score,
        })

    # ✅ SAVE results
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("[INFO] results.json saved")

if __name__ == "__main__":
    main()
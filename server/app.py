"""
Email Triage OpenEnv — FastAPI Server
Exposes the OpenEnv HTTP API: /reset, /step, /state, /health
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.environment import EmailTriageEnv, TASK_NAMES
from server.models import EmailObservation, TriageAction, StepResult, EpisodeState


# ── Session store (in-memory, single-user for benchmark purposes) ──────────
_sessions: dict[str, EmailTriageEnv] = {}


def get_env(session_id: str = "default") -> EmailTriageEnv:
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found. Call /reset first.")
    return _sessions[session_id]


# ── Request/Response schemas ───────────────────────────────────────────────

class ResetRequest(BaseModel):
    task: str = "priority-classification"
    seed: int | None = None
    session_id: str = "default"


class StepRequest(BaseModel):
    action: dict[str, Any]
    session_id: str = "default"


class ResetResponse(BaseModel):
    observation: dict[str, Any]
    task: str
    session_id: str


class StepResponse(BaseModel):
    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]
    reward_detail: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    tasks: list[str]
    version: str


# ── App ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm a default session
    env = EmailTriageEnv(task_name="priority-classification")
    env.reset()
    _sessions["default"] = env
    yield
    _sessions.clear()


app = FastAPI(
    title="Email Triage OpenEnv",
    description="OpenEnv environment for email prioritization, categorization, and routing.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", tasks=TASK_NAMES, version="1.0.0")


@app.post("/reset", response_model=ResetResponse)
async def reset(req: ResetRequest = ResetRequest()):
    task = req.task if req.task else "priority-classification"
    if task not in TASK_NAMES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task: {task!r}. Valid tasks: {TASK_NAMES}",
        )
    env = EmailTriageEnv(task_name=task, seed=req.seed)
    obs = env.reset()
    _sessions[req.session_id] = env
    return ResetResponse(
        observation=obs.model_dump(),
        task=task,
        session_id=req.session_id,
    )


@app.post("/step", response_model=StepResponse)
async def step(req: StepRequest):
    env = get_env(req.session_id)
    try:
        result = env.step(req.action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=result.observation.model_dump(),
        reward=result.reward,
        done=result.done,
        info=result.info,
        reward_detail=result.reward_detail.model_dump() if result.reward_detail else None,
    )


@app.get("/state", response_model=dict)
async def state(session_id: str = Query(default="default")):
    env = get_env(session_id)
    return env.state().model_dump()


@app.get("/tasks")
async def list_tasks():
    """List all available tasks with descriptions."""
    return {
        "tasks": [
            {
                "name": "priority-classification",
                "difficulty": "easy",
                "description": "Classify email priority (critical/high/medium/low). Graded on priority accuracy with partial credit via adjacency matrix.",
                "max_steps": 10,
                "fields_graded": ["priority", "sentiment"],
            },
            {
                "name": "category-routing",
                "difficulty": "medium",
                "description": "Classify priority + category + route to team + extract action items + flag risks.",
                "max_steps": 15,
                "fields_graded": ["priority", "category", "route_to", "action_items", "flags"],
            },
            {
                "name": "full-triage-pipeline",
                "difficulty": "hard",
                "description": "Full pipeline: all fields including SLA hours, sentiment, PII/compliance flags, detailed action items.",
                "max_steps": 20,
                "fields_graded": ["priority", "category", "route_to", "action_items", "sla_hours", "sentiment", "flags"],
            },
        ]
    }


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()

"""
Email Triage OpenEnv — Core Environment Logic
Implements step() / reset() / state() per the OpenEnv spec.
"""

import random
import sys
import os
from typing import Any

# Allow importing from parent and data directories
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.models import EmailObservation, TriageAction, StepResult, EpisodeState
from server.graders import grade, TASK_MAX_STEPS
from data.emails import get_emails_for_task, get_random_email


TASK_NAMES = ["priority-classification", "category-routing", "full-triage-pipeline"]


class EmailTriageEnv:
    """
    Email Triage Environment implementing the full OpenEnv interface.

    Supports three tasks:
      - priority-classification (easy)
      - category-routing (medium)
      - full-triage-pipeline (hard)
    """

    def __init__(self, task_name: str = "priority-classification", seed: int | None = None):
        if task_name not in TASK_NAMES:
            raise ValueError(f"Unknown task: {task_name!r}. Choose from {TASK_NAMES}")
        self.task_name = task_name
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._best_score = 0.0
        self._action_history: list[dict[str, Any]] = []
        self._current_email: dict[str, Any] = {}
        self._email_queue: list[dict[str, Any]] = []
        self._email_index = 0

    # ── OpenEnv Interface ────────────────────────────────────────────────────

    def reset(self) -> EmailObservation:
        """Reset the environment and return the first observation."""
        self._step = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._best_score = 0.0
        self._action_history = []

        # Shuffle emails for this episode
        emails = get_emails_for_task(self.task_name)
        if not emails:
            raise RuntimeError(f"No emails available for task {self.task_name!r}")
        self._email_queue = emails.copy()
        random.shuffle(self._email_queue)
        self._email_index = 0
        self._current_email = self._email_queue[self._email_index]

        return self._make_observation(feedback="")

    def step(self, action: dict[str, Any] | TriageAction) -> StepResult:
        """
        Submit a triage decision and receive reward + next observation.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        if isinstance(action, TriageAction):
            action_dict = action.model_dump()
        else:
            action_dict = dict(action)

        self._step += 1
        ground_truth = self._current_email.get("ground_truth", {})

        # Grade the action
        scores = grade(self.task_name, action_dict, ground_truth)
        reward = scores["total"]
        self._cumulative_reward += reward
        self._best_score = max(self._best_score, reward)

        # Build feedback for the agent
        feedback = self._build_feedback(scores, ground_truth, action_dict)

        # Record action history
        self._action_history.append({
            "step": self._step,
            "action": action_dict,
            "reward": reward,
            "scores": scores,
        })

        # Advance to next email or end episode
        max_steps = TASK_MAX_STEPS.get(self.task_name, 10)
        self._email_index += 1

        if self._email_index >= len(self._email_queue) or self._step >= max_steps:
            self._done = True
            obs = self._make_observation(feedback=feedback)
        else:
            self._current_email = self._email_queue[self._email_index]
            obs = self._make_observation(feedback=feedback)

        from server.models import TriageReward
        reward_detail = TriageReward(**scores)

        return StepResult(
            observation=obs,
            reward=reward,
            done=self._done,
            info={
                "scores": scores,
                "ground_truth": ground_truth,
                "email_id": self._current_email.get("email_id", ""),
                "step": self._step,
                "cumulative_reward": self._cumulative_reward,
            },
            reward_detail=reward_detail,
        )

    def state(self) -> EpisodeState:
        """Return the full current state of the environment."""
        return EpisodeState(
            task_name=self.task_name,
            email_id=self._current_email.get("email_id", ""),
            step=self._step,
            max_steps=TASK_MAX_STEPS.get(self.task_name, 10),
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            best_score=self._best_score,
            current_email={
                k: v for k, v in self._current_email.items()
                if k != "ground_truth"  # Don't leak ground truth in state
            },
            action_history=self._action_history,
            ground_truth=None,  # Hidden from agent
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _make_observation(self, feedback: str) -> EmailObservation:
        email = self._current_email
        return EmailObservation(
            email_id=email.get("email_id", ""),
            subject=email.get("subject", ""),
            sender=email.get("sender", ""),
            sender_domain=email.get("sender_domain", ""),
            body=email.get("body", ""),
            timestamp=email.get("timestamp", ""),
            thread_length=email.get("thread_length", 1),
            has_attachments=email.get("has_attachments", False),
            step=self._step,
            previous_actions=self._action_history[-3:],  # Last 3 actions as context
            feedback=feedback,
            task_name=self.task_name,
        )

    def _build_feedback(
        self,
        scores: dict[str, float],
        ground_truth: dict[str, Any],
        action: dict[str, Any],
    ) -> str:
        """Build human-readable feedback for the agent after each step."""
        lines = [f"Step {self._step} reward: {scores['total']:.3f}"]

        if scores.get("priority_score", 0) < 0.5:
            lines.append(
                f"Priority: your '{action.get('priority')}' vs expected '{ground_truth.get('priority')}'"
            )
        if scores.get("category_score", 0) < 0.5:
            lines.append(
                f"Category: your '{action.get('category')}' vs expected '{ground_truth.get('category')}'"
            )
        if scores.get("routing_score", 0) < 0.5 and "route_to" in ground_truth:
            lines.append(
                f"Routing: your '{action.get('route_to')}' vs expected '{ground_truth.get('route_to')}'"
            )
        if scores.get("penalty", 0) > 0:
            lines.append(f"Penalty applied: {scores['penalty']:.3f} (check flags)")

        return " | ".join(lines)

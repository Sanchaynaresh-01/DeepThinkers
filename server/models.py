"""
Typed Pydantic models for the Email Triage OpenEnv environment.
"""

from enum import Enum
from typing import Any, Optional

try:
    from pydantic import BaseModel, Field
except ImportError:
    class BaseModel:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)
        def model_dump(self):
            return {k: v for k, v in self.__dict__.items()}
    def Field(default=None, **_):
        return default


class Priority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    SUPPORT = "support"
    BILLING = "billing"
    SALES = "sales"
    LEGAL = "legal"
    HR = "hr"
    ENGINEERING = "engineering"
    SPAM = "spam"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    URGENT = "urgent"


class EmailObservation(BaseModel):
    email_id: str = ""
    subject: str = ""
    sender: str = ""
    sender_domain: str = ""
    body: str = ""
    timestamp: str = ""
    thread_length: int = 1
    has_attachments: bool = False
    step: int = 0
    previous_actions: list = Field(default_factory=list)
    feedback: str = ""
    task_name: str = ""


class TriageAction(BaseModel):
    priority: Optional[str] = None
    category: Optional[str] = None
    route_to: Optional[str] = None
    action_items: list = Field(default_factory=list)
    sla_hours: Optional[int] = None
    sentiment: Optional[str] = None
    flags: list = Field(default_factory=list)
    reasoning: str = ""


class TriageReward(BaseModel):
    total: float = 0.0
    priority_score: float = 0.0
    category_score: float = 0.0
    routing_score: float = 0.0
    action_items_score: float = 0.0
    sla_score: float = 0.0
    sentiment_score: float = 0.0
    flags_score: float = 0.0
    penalty: float = 0.0
    breakdown: dict = Field(default_factory=dict)


class StepResult(BaseModel):
    observation: Any = None
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)
    reward_detail: Any = None


class EpisodeState(BaseModel):
    task_name: str = ""
    email_id: str = ""
    step: int = 0
    max_steps: int = 10
    done: bool = False
    cumulative_reward: float = 0.0
    best_score: float = 0.0
    current_email: dict = Field(default_factory=dict)
    action_history: list = Field(default_factory=list)
    ground_truth: Optional[dict] = None

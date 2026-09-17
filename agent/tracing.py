# agent/tracing.py
import json, time, uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

TRACE_LOG = Path(__file__).resolve().parent.parent / "logs" / "traces.jsonl"

@dataclass
class TraceEvent:
    # identity / context
    trace_id: str
    conversation_id: str
    workflow_id: str                    # which LangGraph run this belongs to
    step_name: str                      # e.g. "intent_classify", "search_medical_history", "gemini_generate"
    event_type: str                     # "tool_call" | "llm_call" | "node_transition" | "user_feedback" | "eval"

    # model / prompt provenance
    model_name: Optional[str] = None    # "gemini-3.6-flash"
    model_version: Optional[str] = None # provider's actual version string, not just your label
    prompt_version: Optional[str] = None # e.g. git hash or semver of the prompt template used

    # tool-specific
    tool_name: Optional[str] = None
    tool_arguments: Optional[dict] = None

    # performance
    latency_ms: Optional[float] = None
    time_to_first_token_ms: Optional[float] = None
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    cost_usd: Optional[float] = None

    # reliability
    retries: int = 0
    fallback_used: Optional[str] = None  # e.g. "gemini_to_mock" if you ever allow it — you said you don't, so this should almost always be None/False
    error: Optional[str] = None

    # feedback / quality
    user_feedback: Optional[str] = None  # "up" | "down" | None
    eval_scores: Optional[dict] = None   # {"intent_accuracy": 1, "refusal_correct": 1} — filled in by offline eval runs

    payload: dict = field(default_factory=dict)  # anything event-specific that doesn't fit above
    timestamp: float = field(default_factory=time.time)

def log_event(**kwargs) -> str:
    event = TraceEvent(trace_id=str(uuid.uuid4()), **kwargs)
    TRACE_LOG.parent.mkdir(exist_ok=True)
    with open(TRACE_LOG, "a") as f:
        f.write(json.dumps(asdict(event)) + "\n")
    return event.trace_id

def read_events(trace_file=None):
    """
    Read raw trace events from the JSONL trace file.

    Returns:
        Iterator of dictionaries, one per trace event.
    """
    path = trace_file or TRACE_LOG

    if not path.exists():
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Ignore malformed lines instead of crashing the dashboard
                continue

def log_issue(conversation_id: str | None, workflow_id: str | None,
             issue_type: str, step_name: str = "observability", payload: dict | None = None) -> str:
    """Record a normalized observability issue without affecting workflow execution."""
    return log_event(
        conversation_id=conversation_id or "untagged",
        workflow_id=workflow_id or "untagged",
        step_name=step_name,
        event_type=issue_type,
        payload=payload or {},
    )


def log_eval(conversation_id: str | None, workflow_id: str | None,
             actual_outcome: str, expected_outcome: str, payload: dict | None = None) -> str:
    """Record one offline-evaluation outcome for final-outcome accuracy."""
    data = dict(payload or {})
    data.update({"actual_outcome": actual_outcome, "expected_outcome": expected_outcome})
    return log_event(
        conversation_id=conversation_id or "untagged",
        workflow_id=workflow_id or "untagged",
        step_name="offline_eval",
        event_type="eval",
        payload=data,
    )


def log_hitl_rejection(conversation_id: str | None, workflow_id: str | None,
                       reason: str | None = None, payload: dict | None = None) -> str:
    """Record a human-in-the-loop rejection as a first-class trace event."""
    data = dict(payload or {})
    if reason:
        data["reason"] = reason
    return log_event(
        conversation_id=conversation_id or "untagged",
        workflow_id=workflow_id or "untagged",
        step_name="hitl",
        event_type="hitl_rejection",
        payload=data,
    )

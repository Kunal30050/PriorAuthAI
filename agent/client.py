import asyncio
import json
import threading
import time
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from tracing import log_event
except ImportError:
    from agent.tracing import log_event

import sys
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "agent.mcp_tools.server"],
)

# Lightweight schemas for observability. Validation is telemetry-only: an
# invalid call is still sent to MCP so instrumentation cannot change behavior.
TOOL_SCHEMAS = {
    "health_check": {},
    "get_patient": {"patient_id": str},
    "get_authorization_history": {"patient_id": str},
    "get_relevant_authorization_history": {"patient_id": str, "procedure_code": str},
    "get_insurance_requirements": {"insurance_id": str, "procedure_code": str},
    "check_documents": {"patient_id": str, "required_documents": list},
    "check_document_status": {"document_status": dict, "required_documents": list},
}


# In-memory record of tool calls made per workflow_id, used only to
# detect redundant repeat calls within one workflow run. This avoids
# re-reading and re-parsing the entire (ever-growing) trace log from
# disk on every single tool call, which got slower over time as
# logs/traces.jsonl grew. Cleared lazily -- entries older than a few
# minutes are dropped so this dict can't grow unbounded across a
# long-running server process.
_recent_calls_lock = threading.Lock()
_recent_calls: dict[str, set[str]] = {}
_recent_calls_seen_at: dict[str, float] = {}
_RECENT_CALLS_TTL_SECONDS = 600


def _prune_recent_calls() -> None:
    cutoff = time.time() - _RECENT_CALLS_TTL_SECONDS
    stale = [
        wid
        for wid, seen_at in _recent_calls_seen_at.items()
        if seen_at < cutoff
    ]
    for wid in stale:
        _recent_calls.pop(wid, None)
        _recent_calls_seen_at.pop(wid, None)


def _check_and_record_call(workflow_id: str, signature: str) -> bool:
    """Returns True if this exact (tool, arguments) signature was already
    called earlier in this workflow_id. O(1) in-memory, no disk I/O."""
    with _recent_calls_lock:
        _prune_recent_calls()
        calls = _recent_calls.setdefault(workflow_id, set())
        _recent_calls_seen_at[workflow_id] = time.time()
        was_seen = signature in calls
        calls.add(signature)
        return was_seen


def _validate_args(tool_name: str, arguments: dict) -> tuple[bool, list[str]]:
    schema = TOOL_SCHEMAS.get(tool_name)
    if schema is None:
        return False, [f"unknown_tool:{tool_name}"]
    if not isinstance(arguments, dict):
        return False, ["arguments_not_object"]
    errors = []
    for key, typ in schema.items():
        if key not in arguments:
            errors.append(f"missing:{key}")
        elif not isinstance(arguments[key], typ):
            errors.append(f"type:{key}:expected_{typ.__name__}")
        elif isinstance(arguments[key], str) and not arguments[key].strip():
            errors.append(f"empty:{key}")
    for key in arguments:
        if key not in schema:
            errors.append(f"unexpected:{key}")
    return not errors, errors


def _result_dict(result: Any) -> dict | None:
    try:
        content = getattr(result, "content", [])
        if not content:
            return None
        text = getattr(content[0], "text", None)
        if not text:
            return None
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _retrieval_failure(tool_name: str, value: dict | None) -> bool:
    if tool_name not in {"get_authorization_history", "get_relevant_authorization_history"}:
        return False
    return value is None


async def call_mcp_tool(tool_name: str, arguments: dict):
    async with stdio_client(SERVER_PARAMS) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            return await session.call_tool(tool_name, arguments)


def call_mcp_tool_sync(
    tool_name: str,
    arguments: dict,
    conversation_id: str | None = None,
    workflow_id: str | None = None,
    expected_tool: str | None = None,
):
    start = time.time()
    error = None
    valid, schema_errors = _validate_args(tool_name, arguments)
    cid = conversation_id or "untagged"
    wid = workflow_id or "untagged"

    try:
        log_event(
            conversation_id=cid,
            workflow_id=wid,
            step_name=tool_name,
            event_type="schema_check",
            tool_name=tool_name,
            tool_arguments=arguments,
            payload={
                "valid": valid,
                "errors": schema_errors,
                "expected_tool": expected_tool,
            },
        )
    except Exception:
        pass

    if expected_tool and expected_tool != tool_name:
        try:
            log_event(
                conversation_id=cid,
                workflow_id=wid,
                step_name=tool_name,
                event_type="tool_selection_error",
                tool_name=tool_name,
                tool_arguments=arguments,
                payload={"expected_tool": expected_tool, "wrong_tool": True},
            )
        except Exception:
            pass

    try:
        result = asyncio.run(call_mcp_tool(tool_name, arguments))
        parsed = _result_dict(result)
        return result
    except Exception as exc:
        error = str(exc)
        raise
    finally:
        try:
            # Detect repeated identical calls inside one workflow. Uses an
            # in-memory record (see _check_and_record_call) instead of
            # re-reading the whole trace log from disk on every call.
            redundant = False
            if wid != "untagged":
                signature = json.dumps([tool_name, arguments], sort_keys=True, default=str)
                redundant = _check_and_record_call(wid, signature)
            retrieval_failure = _retrieval_failure(
                tool_name,
                parsed if error is None else None,
            )
            retrieval_empty = (
                tool_name in {
                    "get_authorization_history",
                    "get_relevant_authorization_history",
                }
                and error is None
                and parsed is not None
                and parsed.get("history_count") == 0
                and parsed.get("history") == []
            )
            payload = {
                "expected_tool": expected_tool,
                "wrong_tool": bool(expected_tool and expected_tool != tool_name),
                "retrieval_failure": retrieval_failure,
                "retrieval_empty": retrieval_empty,
                "redundant_call": redundant,
            }
            success = error is None and parsed is not None

            payload = {
                "expected_tool": expected_tool,
                "wrong_tool": bool(expected_tool and expected_tool != tool_name),
                "success": success,
                "mcp_error": error is not None,
                "valid_payload": parsed is not None,
                "retrieval_failure": retrieval_failure,
                "retrieval_empty": retrieval_empty,
                "redundant_call": redundant,
            }
            log_event(
                conversation_id=cid,
                workflow_id=wid,
                step_name=tool_name,
                event_type="tool_call",
                tool_name=tool_name,
                tool_arguments=arguments,
                latency_ms=(time.time() - start) * 1000,
                error=error,
                payload=payload,
            )
            
        except Exception:
            # Telemetry must never break the actual MCP workflow.
            pass

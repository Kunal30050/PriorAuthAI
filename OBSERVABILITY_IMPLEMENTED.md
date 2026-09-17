# PriorAuthAI observability implementation

This patch keeps tracing and metrics separate: `tracing.py` writes raw events; `metrics.py` derives health/failure metrics.

## Detectors now supported

- Wrong tool: compares `expected_tool` with the actual MCP tool and records `tool_selection_error` / `wrong_tool`.
- Wrong arguments: lightweight per-tool schema validation emits `schema_check`; invalid schemas become `wrong_argument_calls`.
- Retrieval failure: history retrieval returning an empty history (or failing) is flagged for history tools.
- Agent loop: repeated LangGraph node in one workflow is flagged by the node wrapper.
- Redundant calls: identical MCP tool + argument calls repeated inside one workflow are flagged.
- Invalid state transition: traced node sequence is compared with the declared graph predecessor.
- Guardrail violation: generated output containing approval/denial language is flagged (while allowing explicit "do not approve/deny" wording).
- API failure: MCP exceptions are captured on `tool_call` events.
- Latency spike: values above `max(2x mean, mean + 3 sigma)` are counted when at least three samples exist.
- Cost anomaly: same relative threshold for LLM call costs when at least three cost samples exist.
- HITL rejection: `tracing.log_hitl_rejection(...)` records a first-class event; metrics count it.
- Incorrect final outcome: offline `eval` events with `actual_outcome` and `expected_outcome` are compared.

## Important honesty boundary

The wrong-tool detector is semantic only when the caller supplies `expected_tool`. The bundled workflow does this for its MCP calls. Retrieval detection is deliberately limited to tools whose response has an explicit history count, avoiding guesses about arbitrary tools.

No MCP server behavior is changed by this patch. Argument validation is telemetry-only and does not block calls.

"""
Derived observability metrics for PriorAuthAI.

Raw events are written by tracing.py. This module is read-only and derives
health, failure-mode, anomaly, and offline-evaluation metrics from them.
"""

from collections import defaultdict
from datetime import datetime, timezone
from statistics import mean, pstdev


# ---------------------------------------------------------------------------
# BASIC HELPERS
# ---------------------------------------------------------------------------

def _day_bucket(ts: float) -> str:
    return datetime.fromtimestamp(
        ts,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d")


def _pctl(values, q):
    if not values:
        return None

    values = sorted(values)

    idx = min(
        len(values) - 1,
        max(
            0,
            int(
                round(
                    (len(values) - 1) * q
                )
            ),
        ),
    )

    return values[idx]


def _event_payload(e):
    payload = e.get("payload")

    return (
        payload
        if isinstance(payload, dict)
        else {}
    )


# ---------------------------------------------------------------------------
# WORKFLOW OUTCOME HELPERS
# ---------------------------------------------------------------------------

def _workflow_outcomes(events):
    """
    Return terminal workflow outcomes.

    Expected event shape:

        event_type = "workflow_outcome"

        payload = {
            "outcome": "COMPLETED" | "ESCALATED" | "FAILED",
            ...
        }

    We intentionally use only workflow_outcome events here.

    Node transitions are execution telemetry, not terminal task outcomes.
    """

    outcomes = []

    for event in events:

        if event.get("event_type") != "workflow_outcome":
            continue

        payload = _event_payload(event)

        outcome = payload.get("outcome")

        if outcome not in {
            "COMPLETED",
            "ESCALATED",
            "FAILED",
        }:
            continue

        outcomes.append(event)

    return outcomes


# ---------------------------------------------------------------------------
# ANOMALIES / FAILURE DETECTION
# ---------------------------------------------------------------------------

def _anomalies(
    events,
    tool_calls,
    llm_calls,
):
    """
    Return explicit failure/anomaly counters.

    Detectors only count evidence present in traces; they never manufacture a
    failure. Semantic wrong-tool detection is based on the optional
    expected_tool field recorded by the workflow/client.
    """

    wrong_tool = 0
    wrong_arguments = 0
    retrieval_failure = 0
    agent_loop = 0
    redundant_calls = 0
    invalid_transition = 0
    guardrail = 0
    api_failure = 0
    hitl_rejection = 0
    incorrect_outcome = 0

    for e in events:

        payload = _event_payload(e)
        event_type = e.get("event_type")

        # ---------------------------------------------------------------
        # Schema validation
        # ---------------------------------------------------------------

        if (
            event_type == "schema_check"
            and not payload.get("valid", True)
        ):
            wrong_arguments += 1

        # ---------------------------------------------------------------
        # MCP/tool telemetry
        # ---------------------------------------------------------------

        if event_type == "tool_call":

            if (
                payload.get("wrong_tool")
                or (
                    payload.get("expected_tool")
                    and payload.get("expected_tool")
                    != e.get("tool_name")
                )
            ):
                wrong_tool += 1

            if e.get("error"):
                api_failure += 1

            if payload.get("retrieval_failure"):
                retrieval_failure += 1

            if payload.get("redundant_call"):
                redundant_calls += 1

        # ---------------------------------------------------------------
        # LangGraph node telemetry
        # ---------------------------------------------------------------

        elif event_type == "node_transition":

            if payload.get("agent_loop"):
                agent_loop += 1

            if payload.get(
                "invalid_state_transition"
            ):
                invalid_transition += 1

        # ---------------------------------------------------------------
        # Guardrails
        # ---------------------------------------------------------------

        elif event_type == "guardrail_violation":

            guardrail += 1

        # ---------------------------------------------------------------
        # HITL
        # ---------------------------------------------------------------

        elif event_type == "hitl_rejection":

            hitl_rejection += 1

        # ---------------------------------------------------------------
        # Offline evaluation
        # ---------------------------------------------------------------

        elif (
            event_type == "eval"
            and payload.get(
                "expected_outcome"
            ) is not None
        ):

            if (
                payload.get("actual_outcome")
                != payload.get("expected_outcome")
            ):
                incorrect_outcome += 1

        # ---------------------------------------------------------------
        # User feedback
        # ---------------------------------------------------------------

        if (
            event_type == "user_feedback"
            and e.get("user_feedback") == "down"
            and payload.get("hitl")
        ):
            hitl_rejection += 1

    # -----------------------------------------------------------------------
    # LATENCY ANOMALIES
    # -----------------------------------------------------------------------

    latencies = [
        e.get("latency_ms")
        for e in tool_calls + llm_calls
        if e.get("latency_ms") is not None
    ]

    spike_threshold = None
    latency_spikes = 0

    if len(latencies) >= 3:

        mu = mean(latencies)
        sd = pstdev(latencies)

        spike_threshold = max(
            mu * 2.0,
            mu + 3.0 * sd,
        )

        latency_spikes = sum(
            1
            for value in latencies
            if value > spike_threshold
        )

    # -----------------------------------------------------------------------
    # COST ANOMALIES
    # -----------------------------------------------------------------------

    costs = [
        e.get("cost_usd")
        for e in llm_calls
        if e.get("cost_usd") is not None
    ]

    cost_anomalies = 0
    cost_threshold = None

    if len(costs) >= 3:

        mu = mean(costs)
        sd = pstdev(costs)

        cost_threshold = max(
            mu * 2.0,
            mu + 3.0 * sd,
        )

        cost_anomalies = sum(
            1
            for value in costs
            if value > cost_threshold
        )

    return {
        "wrong_tool_calls": wrong_tool,
        "wrong_argument_calls": wrong_arguments,
        "retrieval_failures": retrieval_failure,
        "agent_loops": agent_loop,
        "redundant_tool_calls": redundant_calls,
        "invalid_state_transitions": invalid_transition,
        "guardrail_violations": guardrail,
        "api_failures": api_failure,
        "latency_spikes": latency_spikes,
        "latency_spike_threshold_ms": spike_threshold,
        "cost_anomalies": cost_anomalies,
        "cost_anomaly_threshold_usd": cost_threshold,
        "hitl_rejections": hitl_rejection,
        "incorrect_final_outcomes": incorrect_outcome,
    }


# ---------------------------------------------------------------------------
# MAIN METRICS
# ---------------------------------------------------------------------------

def compute_all_metrics(
    events: list[dict],
) -> dict:

    by_type = defaultdict(list)

    for event in events:

        event_type = event.get(
            "event_type"
        )

        if event_type:
            by_type[event_type].append(event)

    # -----------------------------------------------------------------------
    # EVENT GROUPS
    # -----------------------------------------------------------------------

    tool_calls = by_type["tool_call"]

    schema_checks = by_type[
        "schema_check"
    ]

    node_transitions = by_type[
        "node_transition"
    ]

    llm_calls = by_type[
        "llm_call"
    ]

    feedback = by_type[
        "user_feedback"
    ]

    eval_events = by_type[
        "eval"
    ]

    quality_eval_events = by_type[
        "quality_eval"
    ]

    workflow_outcomes = _workflow_outcomes(
        events
    )

    # -----------------------------------------------------------------------
    # ROOT RESULT OBJECT
    # -----------------------------------------------------------------------

    result = {
        "generated_at": (
            datetime.now(
                tz=timezone.utc
            ).isoformat()
        ),
        "summary": {},
        "trend": {},
        "eval": {},
        "llm_eval": {},
        "failures": {},
        "anomalies": {},
    }

    # =======================================================================
    # 1. TOOL CALL SUCCESS RATE
    # =======================================================================

    successes = sum(
        1
        for event in tool_calls
        if (
            event.get("payload", {}).get(
                "success"
            ) is True
        )
        or (
            # Backward compatibility for old traces
            "success"
            not in event.get("payload", {})
            and not event.get("error")
        )
    )

    result["summary"][
        "tool_call_success_rate"
    ] = (
        successes / len(tool_calls)
        if tool_calls
        else None
    )

    result["summary"][
        "tool_call_count"
    ] = len(tool_calls)

    # =======================================================================
    # 2. INVALID SCHEMA RATE
    # =======================================================================

    invalid = sum(
        1
        for event in schema_checks
        if not _event_payload(event).get(
            "valid",
            True,
        )
    )

    result["summary"][
        "invalid_schema_rate"
    ] = (
        invalid / len(schema_checks)
        if schema_checks
        else None
    )

    result["summary"][
        "schema_check_count"
    ] = len(schema_checks)

    # =======================================================================
    # 3. WORKFLOW OUTCOME METRICS
    # =======================================================================

    completed = sum(
        1
        for event in workflow_outcomes
        if _event_payload(event).get(
            "outcome"
        ) == "COMPLETED"
    )

    escalated = sum(
        1
        for event in workflow_outcomes
        if _event_payload(event).get(
            "outcome"
        ) == "ESCALATED"
    )

    failed = sum(
        1
        for event in workflow_outcomes
        if _event_payload(event).get(
            "outcome"
        ) == "FAILED"
    )

    total_workflows = len(
        workflow_outcomes
    )

    result["summary"][
        "task_completion_rate"
    ] = (
        completed / total_workflows
        if total_workflows
        else None
    )

    result["summary"][
        "escalation_rate"
    ] = (
        escalated / total_workflows
        if total_workflows
        else None
    )

    result["summary"][
        "workflow_failure_rate"
    ] = (
        failed / total_workflows
        if total_workflows
        else None
    )

    result["summary"][
        "workflow_count"
    ] = total_workflows

    result["summary"][
        "completions_count"
    ] = completed

    result["summary"][
        "escalations_count"
    ] = escalated

    result["summary"][
        "failures_count"
    ] = failed

    # =======================================================================
    # 4. TOOL RETRY / FALLBACK METRICS
    # =======================================================================

    total_retries = sum(
        event.get(
            "retries",
            0,
        )
        for event in tool_calls
    )

    result["summary"][
        "avg_retries_per_tool_call"
    ] = (
        total_retries / len(tool_calls)
        if tool_calls
        else None
    )

    result["summary"][
        "fallback_calls"
    ] = sum(
        1
        for event in llm_calls
        if event.get(
            "fallback_used"
        )
    )

    # =======================================================================
    # 5. COST METRICS
    # =======================================================================

    total_cost = sum(
        event.get(
            "cost_usd"
        )
        or 0
        for event in llm_calls
    )

    result["summary"][
        "total_cost_usd"
    ] = total_cost

    result["summary"][
        "cost_per_successful_task"
    ] = (
        total_cost / completed
        if completed
        else None
    )

    # =======================================================================
    # 6. USER FEEDBACK
    # =======================================================================

    down = sum(
        1
        for event in feedback
        if event.get(
            "user_feedback"
        ) == "down"
    )

    result["summary"][
        "user_flag_rate"
    ] = (
        down / len(feedback)
        if feedback
        else None
    )

    result["summary"][
        "feedback_count"
    ] = len(feedback)

    # =======================================================================
    # 7. LATENCY
    # =======================================================================

    latencies = [
        event["latency_ms"]
        for event in (
            tool_calls + llm_calls
        )
        if event.get(
            "latency_ms"
        ) is not None
    ]

    result["summary"][
        "latency_p50_ms"
    ] = _pctl(
        latencies,
        0.50,
    )

    result["summary"][
        "latency_p95_ms"
    ] = _pctl(
        latencies,
        0.95,
    )

    # =======================================================================
    # 8. FAILURE / ANOMALY DIAGNOSTICS
    # =======================================================================

    diagnostics = _anomalies(
        events,
        tool_calls,
        llm_calls,
    )

    result["failures"] = {
        key: diagnostics[key]
        for key in (
            "wrong_tool_calls",
            "wrong_argument_calls",
            "retrieval_failures",
            "agent_loops",
            "redundant_tool_calls",
            "invalid_state_transitions",
            "guardrail_violations",
            "api_failures",
            "hitl_rejections",
            "incorrect_final_outcomes",
        )
    }

    result["anomalies"] = {
        key: diagnostics[key]
        for key in (
            "latency_spikes",
            "latency_spike_threshold_ms",
            "cost_anomalies",
            "cost_anomaly_threshold_usd",
        )
    }

    # =======================================================================
    # 9. DAILY TRENDS
    # =======================================================================

    cost_by_day = defaultdict(float)
    completions_by_day = defaultdict(int)
    escalations_by_day = defaultdict(int)
    failures_by_day = defaultdict(int)
    latency_by_day = defaultdict(list)

    # -----------------------------------------------------------------------
    # Cost trend
    # -----------------------------------------------------------------------

    for event in llm_calls:

        timestamp = event.get(
            "timestamp"
        )

        if timestamp is None:
            continue

        cost_by_day[
            _day_bucket(timestamp)
        ] += (
            event.get(
                "cost_usd"
            )
            or 0
        )

    # -----------------------------------------------------------------------
    # Workflow outcome trends
    # -----------------------------------------------------------------------

    for event in workflow_outcomes:

        timestamp = event.get(
            "timestamp"
        )

        if timestamp is None:
            continue

        day = _day_bucket(
            timestamp
        )

        outcome = _event_payload(
            event
        ).get("outcome")

        if outcome == "COMPLETED":

            completions_by_day[
                day
            ] += 1

        elif outcome == "ESCALATED":

            escalations_by_day[
                day
            ] += 1

        elif outcome == "FAILED":

            failures_by_day[
                day
            ] += 1

    # -----------------------------------------------------------------------
    # Latency trends
    # -----------------------------------------------------------------------

    for event in (
        tool_calls + llm_calls
    ):

        timestamp = event.get(
            "timestamp"
        )

        latency = event.get(
            "latency_ms"
        )

        if (
            timestamp is None
            or latency is None
        ):
            continue

        latency_by_day[
            _day_bucket(timestamp)
        ].append(
            latency
        )

    days = sorted(
        set(cost_by_day)
        | set(completions_by_day)
        | set(escalations_by_day)
        | set(failures_by_day)
        | set(latency_by_day)
    )

    result["trend"] = {
        "days": days,

        "cost_usd": [
            round(
                cost_by_day.get(
                    day,
                    0,
                ),
                4,
            )
            for day in days
        ],

        "completions": [
            completions_by_day.get(
                day,
                0,
            )
            for day in days
        ],

        "escalations": [
            escalations_by_day.get(
                day,
                0,
            )
            for day in days
        ],

        "failures": [
            failures_by_day.get(
                day,
                0,
            )
            for day in days
        ],

        "latency_p50_ms": [
            _pctl(
                latency_by_day[day],
                0.50,
            )
            for day in days
        ],
    }

    # =======================================================================
    # 10. OFFLINE EVALUATION
    # =======================================================================

    if eval_events:

        intent_matches = [
            payload.get(
                "actual_intent"
            )
            == payload.get(
                "expected_intent"
            )
            for event in eval_events
            for payload in [
                _event_payload(event)
            ]
            if payload.get(
                "actual_intent"
            ) is not None
        ]

        refusal_matches = [
            payload.get(
                "actual_refusal"
            )
            == payload.get(
                "expected_refusal"
            )
            for event in eval_events
            for payload in [
                _event_payload(event)
            ]
            if payload.get(
                "actual_refusal"
            ) is not None
        ]

        applicable = [
            payload
            for event in eval_events
            for payload in [
                _event_payload(event)
            ]
            if payload.get(
                "expected_retrieval_keywords"
            )
        ]

        hits = [
            any(
                keyword.lower()
                in payload.get(
                    "retrieved_text",
                    "",
                ).lower()
                for keyword in payload[
                    "expected_retrieval_keywords"
                ]
            )
            for payload in applicable
        ]

        result["eval"] = {
            "intent_accuracy": (
                mean(intent_matches)
                if intent_matches
                else None
            ),

            "refusal_accuracy": (
                mean(refusal_matches)
                if refusal_matches
                else None
            ),

            "retrieval_hit_rate": (
                mean(hits)
                if hits
                else None
            ),

            "example_count": len(
                eval_events
            ),
        }

    else:

        result["eval"] = {
            "intent_accuracy": None,
            "refusal_accuracy": None,
            "retrieval_hit_rate": None,
            "example_count": 0,
        }

    # =======================================================================
    # 11. LLM QUALITY EVALUATION (hallucination, faithfulness, tool
    #     selection, decision accuracy, guardrails, etc.)
    #
    #     Each "quality_eval" trace event carries a subset of these
    #     fields depending on its target ("chat" vs "workflow_decision")
    #     -- fields that don't apply to a given event are None and are
    #     excluded from that field's average, never treated as 0.
    # =======================================================================

    def _score_values(field_name: str) -> list[float]:
        values = []
        for event in quality_eval_events:
            payload = _event_payload(event)
            value = payload.get(field_name)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    guardrail_flags = [
        bool(_event_payload(event).get("guardrail_violation"))
        for event in quality_eval_events
    ]

    if quality_eval_events:

        result["llm_eval"] = {
            "hallucination_rate": (
                mean(_score_values("hallucination_rate"))
                if _score_values("hallucination_rate")
                else None
            ),
            "faithfulness": (
                mean(_score_values("faithfulness"))
                if _score_values("faithfulness")
                else None
            ),
            "system_prompt_adherence": (
                mean(_score_values("system_prompt_adherence"))
                if _score_values("system_prompt_adherence")
                else None
            ),
            "goal_achievement": (
                mean(_score_values("goal_achievement"))
                if _score_values("goal_achievement")
                else None
            ),
            "tool_selection_accuracy": (
                mean(_score_values("tool_selection_accuracy"))
                if _score_values("tool_selection_accuracy")
                else None
            ),
            "tool_usage_efficiency": (
                mean(_score_values("tool_usage_efficiency"))
                if _score_values("tool_usage_efficiency")
                else None
            ),
            "decision_accuracy": (
                mean(_score_values("decision_accuracy"))
                if _score_values("decision_accuracy")
                else None
            ),
            "escalation_accuracy": (
                mean(_score_values("escalation_accuracy"))
                if _score_values("escalation_accuracy")
                else None
            ),
            "guardrail_violation_rate": (
                mean([1.0 if f else 0.0 for f in guardrail_flags])
                if guardrail_flags
                else None
            ),
            "guardrail_violation_count": sum(
                1 for f in guardrail_flags if f
            ),
            "example_count": len(quality_eval_events),
            "judged_example_count": sum(
                1
                for event in quality_eval_events
                if _event_payload(event).get("judge_available")
            ),
        }

    else:

        result["llm_eval"] = {
            "hallucination_rate": None,
            "faithfulness": None,
            "system_prompt_adherence": None,
            "goal_achievement": None,
            "tool_selection_accuracy": None,
            "tool_usage_efficiency": None,
            "decision_accuracy": None,
            "escalation_accuracy": None,
            "guardrail_violation_rate": None,
            "guardrail_violation_count": 0,
            "example_count": 0,
            "judged_example_count": 0,
        }

    return result
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parent
TRACE_FILE = ROOT / "logs" / "traces.jsonl"

# Make sure `agent` can be imported
sys.path.insert(0, str(ROOT))
from agent.tracing import read_events
from agent.metrics import compute_all_metrics


# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------

st.set_page_config(
    page_title="PriorAuthAI Observability",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    .stApp {
        background-color: #0f1720;
    }

    section[data-testid="stSidebar"] {
        background-color: #111923;
        border-right: 1px solid #263241;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    h1, h2, h3, h4, p, label {
        color: #e6edf3 !important;
    }

    .metric-card {
        background: #161f2b;
        border: 1px solid #263241;
        padding: 20px;
        min-height: 120px;
    }

    .metric-label {
        color: #8b98a8;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }

    .metric-value {
        color: #e6edf3;
        font-size: 30px;
        font-weight: 700;
        font-family: monospace;
    }

    .metric-sub {
        color: #718096;
        font-size: 12px;
        margin-top: 5px;
    }

    .status-ok {
        color: #2dd4bf;
        font-weight: 700;
    }

    .status-warning {
        color: #f0a93a;
        font-weight: 700;
    }

    .status-error {
        color: #e5484d;
        font-weight: 700;
    }

    .section-title {
        color: #e6edf3;
        font-size: 18px;
        font-weight: 600;
        margin-top: 30px;
        margin-bottom: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def load_events() -> list[dict]:
    """Read raw events from traces.jsonl."""
    return list(read_events(TRACE_FILE))


def pct(value):
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def number(value, decimals=1):
    if value is None:
        return "—"

    if decimals == 0:
        return f"{value:,.0f}"

    return f"{value:,.{decimals}f}"


def metric_card(label, value, subtitle=""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

events = load_events()

if not events:
    st.title("PriorAuthAI — Observability")

    st.warning(
        "No trace events found yet. Run the PriorAuthAI workflow first."
    )

    st.code(
        "logs/traces.jsonl",
        language="text",
    )

    st.stop()


metrics = compute_all_metrics(events)

summary = metrics["summary"]
trend = metrics["trend"]
evaluation = metrics["eval"]


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.title("PriorAuthAI")

st.sidebar.caption("Observability Dashboard")

st.sidebar.divider()

auto_refresh = st.sidebar.checkbox(
    "Auto refresh",
    value=False,
)

if st.sidebar.button("Refresh now"):
    st.rerun()

if auto_refresh:
    # Reruns this script every 5s while the checkbox is on, so new
    # trace events written by a workflow run show up without the
    # user having to touch anything.
    st_autorefresh(interval=5000, key="traces_autorefresh")

st.sidebar.divider()

st.sidebar.write(
    f"**Trace events:** {len(events):,}"
)

st.sidebar.write(
    f"**Trace file:** `{TRACE_FILE}`"
)

st.sidebar.divider()

if TRACE_FILE.exists():
    modified = TRACE_FILE.stat().st_mtime
    from datetime import datetime

    last_updated = datetime.fromtimestamp(modified)

    st.sidebar.write(
        f"Last trace: `{last_updated.strftime('%Y-%m-%d %H:%M:%S')}`"
    )


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("PriorAuthAI — Observability")

st.caption(
    "Raw telemetry → derived metrics → operational dashboard"
)

st.divider()


# ---------------------------------------------------------
# HEALTH STATUS
# ---------------------------------------------------------

tool_success = summary.get("tool_call_success_rate")
error_count = sum(
    1 for event in events
    if event.get("error")
)

fallback_count = summary.get("fallback_calls", 0)

if fallback_count > 5 or error_count > 5:
    status = "DEGRADED"
    status_class = "status-warning"
else:
    status = "HEALTHY"
    status_class = "status-ok"

# st.markdown(
#     f"""
#     <div style="
#         background:#161f2b;
#         border:1px solid #263241;
#         padding:14px 18px;
#         margin-bottom:20px;
#     ">
#         SYSTEM STATUS:
#         <span class="{status_class}">
#             ● {status}
#         </span>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )


# ---------------------------------------------------------
# TOP METRICS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Workflow Vitals</div>',
    unsafe_allow_html=True,
)

cols = st.columns(5)

with cols[0]:
    metric_card(
        "Task Completion",
        pct(summary.get("task_completion_rate")),
        f"{summary.get('completions_count', 0)} successful",
    )

with cols[1]:
    metric_card(
        "Escalation",
        pct(summary.get("escalation_rate")),
        f"{summary.get('escalations_count', 0)} escalated",
    )

with cols[2]:
    metric_card(
        "Tool Success",
        pct(tool_success),
        f"{summary.get('tool_call_count', 0)} MCP calls",
    )

with cols[3]:
    metric_card(
        "User Flags",
        pct(summary.get("user_flag_rate")),
        f"{summary.get('feedback_count', 0)} feedback events",
    )

with cols[4]:
    metric_card(
        "Cost / Success",
        (
            "$" + number(
                summary.get("cost_per_successful_task"),
                4,
            )
            if summary.get("cost_per_successful_task") is not None
            else "—"
        ),
        "USD",
    )


# ---------------------------------------------------------
# RELIABILITY
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Reliability</div>',
    unsafe_allow_html=True,
)

cols = st.columns(5)

with cols[0]:
    metric_card(
        "Invalid Schema",
        pct(summary.get("invalid_schema_rate")),
    )

with cols[1]:
    metric_card(
        "Avg Retries",
        number(
            summary.get("avg_retries_per_tool_call"),
            2,
        ),
    )

with cols[2]:
    metric_card(
        "Fallback Calls",
        number(
            summary.get("fallback_calls"),
            0,
        ),
    )

with cols[3]:
    metric_card(
        "P50 Latency",
        (
            f"{number(summary.get('latency_p50_ms'), 0)} ms"
            if summary.get("latency_p50_ms") is not None
            else "—"
        ),
    )

with cols[4]:
    metric_card(
        "P95 Latency",
        (
            f"{number(summary.get('latency_p95_ms'), 0)} ms"
            if summary.get("latency_p95_ms") is not None
            else "—"
        ),
    )


# ---------------------------------------------------------
# TREND DATAFRAME
# ---------------------------------------------------------

days = trend.get("days", [])

if days:

    trend_df = pd.DataFrame(
        {
            "date": days,
            "Completions": trend.get("completions", []),
            "Escalations": trend.get("escalations", []),
            "Latency P50 (ms)": trend.get(
                "latency_p50_ms",
                [],
            ),
            "Cost (USD)": trend.get(
                "cost_usd",
                [],
            ),
        }
    )

    trend_df["date"] = pd.to_datetime(
        trend_df["date"]
    )

    # -----------------------------------------------------
    # WORKFLOW TREND
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Workflow Trend</div>',
        unsafe_allow_html=True,
    )

    chart_df = trend_df.set_index("date")[
        ["Completions", "Escalations"]
    ]

    st.line_chart(
        chart_df,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # LATENCY TREND
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Latency Trend</div>',
        unsafe_allow_html=True,
    )

    latency_df = trend_df.set_index("date")[
        ["Latency P50 (ms)"]
    ]

    st.line_chart(
        latency_df,
        use_container_width=True,
    )

    # -----------------------------------------------------
    # COST TREND
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">Cost Trend</div>',
        unsafe_allow_html=True,
    )

    cost_df = trend_df.set_index("date")[
        ["Cost (USD)"]
    ]

    st.line_chart(
        cost_df,
        use_container_width=True,
    )


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Offline Evaluation</div>',
    unsafe_allow_html=True,
)

eval_cols = st.columns(4)

with eval_cols[0]:
    metric_card(
        "Intent Accuracy",
        pct(evaluation.get("intent_accuracy")),
    )

with eval_cols[1]:
    metric_card(
        "Refusal Accuracy",
        pct(evaluation.get("refusal_accuracy")),
    )

with eval_cols[2]:
    metric_card(
        "Retrieval Hit Rate",
        pct(evaluation.get("retrieval_hit_rate")),
    )

with eval_cols[3]:
    metric_card(
        "Eval Examples",
        number(
            evaluation.get("example_count"),
            0,
        ),
    )


# ---------------------------------------------------------
# LLM QUALITY EVALUATION
# ---------------------------------------------------------

llm_eval = metrics.get("llm_eval", {})

st.markdown(
    '<div class="section-title">LLM Quality Evaluation</div>',
    unsafe_allow_html=True,
)

st.caption(
    f"{number(llm_eval.get('example_count'), 0)} responses scored "
    f"({number(llm_eval.get('judged_example_count'), 0)} by the LLM "
    f"judge; the rest are mock-mode/deterministic-only runs)"
)

quality_row1 = st.columns(4)

with quality_row1[0]:
    metric_card(
        "Hallucination Rate",
        pct(llm_eval.get("hallucination_rate")),
        "lower is better",
    )

with quality_row1[1]:
    metric_card(
        "Faithfulness",
        pct(llm_eval.get("faithfulness")),
    )

with quality_row1[2]:
    metric_card(
        "System Prompt Adherence",
        pct(llm_eval.get("system_prompt_adherence")),
    )

with quality_row1[3]:
    metric_card(
        "Goal Achievement",
        pct(llm_eval.get("goal_achievement")),
    )

quality_row2 = st.columns(4)

with quality_row2[0]:
    metric_card(
        "Tool Selection Accuracy",
        pct(llm_eval.get("tool_selection_accuracy")),
        "chatbot turns only",
    )

with quality_row2[1]:
    metric_card(
        "Tool Usage Efficiency",
        pct(llm_eval.get("tool_usage_efficiency")),
        "chatbot turns only",
    )

with quality_row2[2]:
    metric_card(
        "Decision Accuracy",
        pct(llm_eval.get("decision_accuracy")),
        "workflow runs only",
    )

with quality_row2[3]:
    metric_card(
        "Escalation Accuracy",
        pct(llm_eval.get("escalation_accuracy")),
        "workflow runs only",
    )

quality_row3 = st.columns(4)

with quality_row3[0]:
    metric_card(
        "Guardrail Violation Rate",
        pct(llm_eval.get("guardrail_violation_rate")),
        f"{number(llm_eval.get('guardrail_violation_count'), 0)} violation(s) found",
    )


# ---------------------------------------------------------
# RAW TRACE INSPECTOR
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Trace Inspector</div>',
    unsafe_allow_html=True,
)

event_types = sorted(
    set(event.get("event_type") for event in events)
)

selected_type = st.selectbox(
    "Event type",
    ["All"] + event_types,
)

filtered_events = events

if selected_type != "All":
    filtered_events = [
        event
        for event in events
        if event.get("event_type") == selected_type
    ]

st.caption(
    f"{len(filtered_events):,} events"
)

# Show latest events first
latest = list(reversed(filtered_events[-100:]))

if latest:

    trace_rows = []

    for event in latest:

        trace_rows.append(
            {
                "timestamp": (
                    pd.to_datetime(
                        event.get("timestamp"),
                        unit="s",
                    ).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    if event.get("timestamp")
                    else ""
                ),
                "event": event.get("event_type"),
                "step": event.get("step_name"),
                "tool": event.get("tool_name") or "",
                "model": event.get("model_name") or "",
                "latency_ms": event.get("latency_ms"),
                "tokens_in": event.get("tokens_in"),
                "tokens_out": event.get("tokens_out"),
                "retries": event.get("retries"),
                "error": event.get("error") or "",
                "workflow_id": event.get("workflow_id"),
            }
        )

    trace_df = pd.DataFrame(trace_rows)

    st.dataframe(
        trace_df,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# EVENT BREAKDOWN
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Event Breakdown</div>',
    unsafe_allow_html=True,
)

event_counts = (
    pd.Series(
        [event.get("event_type") for event in events]
    )
    .value_counts()
    .rename_axis("event_type")
    .reset_index(name="count")
)

st.bar_chart(
    event_counts.set_index("event_type"),
    use_container_width=True,
)


# ---------------------------------------------------------
# RAW TRACE DOWNLOAD
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Raw Data</div>',
    unsafe_allow_html=True,
)

if TRACE_FILE.exists():

    with open(
        TRACE_FILE,
        "rb",
    ) as file:

        st.download_button(
            label="Download traces.jsonl",
            data=file,
            file_name="traces.jsonl",
            mime="application/json",
        )


# ---------------------------------------------------------
# AUTO REFRESH
# ---------------------------------------------------------

if auto_refresh:
    st.caption("Auto refresh is on — this page reruns every 5 seconds.")

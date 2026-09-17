from __future__ import annotations

import json
import threading

from langchain.agents import create_agent
from langchain_core.tools import tool

from llm_config import get_chat_model, is_mock
from mcp_tools.client import call_mcp_tool_sync
from eval_judge import evaluate_chat_turn


_AGENT_SYSTEM_PROMPT = (
    "You are a healthcare prior-authorization assistant.\n\n"
    "You are working with the currently selected patient. "
    "The patient ID is already known to the system. "
    "NEVER ask the user to provide the patient ID.\n\n"
    "Use the available tools whenever information must be "
    "retrieved about the patient.\n\n"
    "You may call multiple tools when necessary to answer "
    "a single question.\n\n"
    "For example, if the user asks whether prior authorization "
    "is required, use the prior-authorization requirements tool.\n\n"
    "If the user asks about previous authorization requests, "
    "use the authorization history tool.\n\n"
    "If the user asks a question involving multiple pieces "
    "of information, call all necessary tools and combine "
    "their results into one answer.\n\n"
    "Use ONLY information returned by the tools. "
    "Do not invent medical facts.\n\n"
    "Keep answers concise and easy to understand.\n\n"
    "You are an information and workflow assistant. "
    "Do not independently approve or deny insurance coverage."
)


def _parse_mcp_result(result) -> dict:
    if getattr(result, "is_error", False):
        raise RuntimeError(f"MCP tool error: {result}")

    content = getattr(result, "content", [])
    if not content:
        raise RuntimeError("MCP tool returned no content")

    text = getattr(content[0], "text", None)
    if not text:
        raise RuntimeError("MCP tool returned no text content")

    return json.loads(text)


# -------------------------------------------------------------------
# MCP → LangChain tool wrappers
# -------------------------------------------------------------------

def _get_patient(patient_id: str) -> dict:
    return _parse_mcp_result(
        call_mcp_tool_sync(
            "get_patient",
            {"patient_id": patient_id},
            expected_tool="get_patient",
        )
    )


def _get_history(patient_id: str) -> dict:
    return _parse_mcp_result(
        call_mcp_tool_sync(
            "get_authorization_history",
            {"patient_id": patient_id},
            expected_tool="get_authorization_history",
        )
    )


def _get_insurance(patient: dict) -> dict:
    return _parse_mcp_result(
        call_mcp_tool_sync(
            "get_insurance_requirements",
            {
                "insurance_id": patient.get("insurance_id", ""),
                "procedure_code": patient.get(
                    "requested_procedure_code", ""
                ),
            },
            expected_tool="get_insurance_requirements",
        )
    )


def _create_patient_tools(patient_id: str):
    """
    Create tools scoped to the currently selected patient.

    The user does NOT need to provide the patient ID.
    The selected patient ID is captured here and passed
    automatically to the MCP tools.
    """

    @tool
    def get_patient_information() -> dict:
        """Get the currently selected patient's basic information."""
        return _get_patient(patient_id)

    @tool
    def get_authorization_history() -> dict:
        """Get the currently selected patient's previous authorization requests."""
        return _get_history(patient_id)

    @tool
    def get_prior_authorization_requirements() -> dict:
        """
        Check whether prior authorization is required for the
        currently selected patient's insurance and requested procedure.
        """
        patient = _get_patient(patient_id)

        if patient.get("patient_found") is False:
            return {
                "error": patient.get(
                    "error",
                    "Patient not found.",
                )
            }

        return _get_insurance(patient)

    return [
        get_patient_information,
        get_authorization_history,
        get_prior_authorization_requirements,
    ]


# -------------------------------------------------------------------
# Mock mode
# -------------------------------------------------------------------

def _mock_answer(message: str, patient: dict) -> str:
    """
    Simple deterministic fallback used when LLM_PROVIDER=mock.

    This keeps the existing test environment working while the
    real LangChain agent is used when Gemini is configured.
    """

    message_lower = message.lower()

    if "diagnosis" in message_lower:
        return (
            f"The patient's diagnosis code is "
            f"{patient.get('diagnosis_code', 'not available')}."
        )

    if "insurance" in message_lower or "payer" in message_lower:
        return (
            f"The patient's insurance ID is "
            f"{patient.get('insurance_id', 'not available')}."
        )

    if "procedure" in message_lower:
        return (
            f"The requested procedure code is "
            f"{patient.get('requested_procedure_code', 'not available')}."
        )

    if (
        "prior auth" in message_lower
        or "prior authorization" in message_lower
        or "authorization required" in message_lower
    ):
        insurance = _get_insurance(patient)

        if insurance.get("rule_found"):
            if insurance.get("requires_prior_auth"):
                return (
                    "Yes, prior authorization is required for "
                    "this payer and procedure."
                )

            return (
                "No, prior authorization is not required for "
                "this payer and procedure."
            )

        return (
            "No matching payer rule was found. "
            "Manual payer verification is required."
        )

    if "history" in message_lower or "previous" in message_lower:
        history = _get_history(patient["patient_id"])
        count = history.get("history_count", 0)

        return (
            f"The patient has {count} previous "
            "authorization record(s)."
        )

    return (
        "I can provide information about the selected patient's "
        "diagnosis, insurance, requested procedure, authorization "
        "history, and prior-authorization requirements."
    )


# -------------------------------------------------------------------
# LangChain agent
# -------------------------------------------------------------------

def _run_agent(patient_id: str, message: str) -> str:
    llm = get_chat_model()

    if llm is None:
        raise RuntimeError(
            "LLM is not configured for agent execution."
        )

    tools = _create_patient_tools(patient_id)

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=_AGENT_SYSTEM_PROMPT,
    )

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": message,
                }
            ]
        }
    )

    messages = result.get("messages", [])

    if not messages:
        return "No answer was generated."

    final_message = messages[-1]
    content = getattr(final_message, "content", final_message)

    if isinstance(content, str):
        answer = content.strip()

    elif isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))

            elif hasattr(item, "text"):
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))

            else:
                parts.append(str(item))

        answer = "".join(parts).strip()

    else:
        answer = str(content).strip()

    # -----------------------------------------------------------------
    # Extract the real tool-call trajectory for evaluation: which tools
    # the agent actually chose to call, and what they actually returned
    # (the ground-truth context the answer must stay faithful to).
    # Never allowed to break the chat response.
    # -----------------------------------------------------------------
    try:
        tool_calls: list[dict] = []
        tool_context_parts: list[str] = []

        for msg in messages:
            msg_tool_calls = getattr(msg, "tool_calls", None)
            if msg_tool_calls:
                for tc in msg_tool_calls:
                    tool_calls.append(
                        {
                            "name": tc.get("name"),
                            "args": tc.get("args") or {},
                        }
                    )

            # ToolMessage instances carry the actual tool output text.
            if type(msg).__name__ == "ToolMessage":
                tool_context_parts.append(
                    str(getattr(msg, "content", ""))
                )

        # Runs in a background thread -- makes its own LLM judge call,
        # and the user shouldn't wait on that for a chat response they
        # already have.
        threading.Thread(
            target=evaluate_chat_turn,
            kwargs=dict(
                conversation_id=patient_id,
                workflow_id=f"chat-{patient_id}",
                system_prompt=_AGENT_SYSTEM_PROMPT,
                user_message=message,
                answer=answer,
                tools_available=[t.name for t in tools],
                tool_calls=tool_calls,
                tool_context="\n".join(tool_context_parts),
            ),
            daemon=True,
        ).start()
    except Exception:
        pass

    return answer


# -------------------------------------------------------------------
# Public API used by Flask
# -------------------------------------------------------------------

def chat_with_patient(patient_id: str, message: str) -> dict:
    patient_id = (patient_id or "").strip()
    message = (message or "").strip()

    if not patient_id:
        return {
            "error": "Patient ID is required.",
            "answer": None,
        }

    if not message:
        return {
            "error": "Message is required.",
            "answer": None,
        }

    # Always validate/retrieve the currently selected patient
    # through MCP first.
    patient = _get_patient(patient_id)

    if patient.get("patient_found") is False:
        return {
            "error": patient.get(
                "error",
                "Patient not found.",
            ),
            "answer": None,
        }

    # ---------------------------------------------------------------
    # Mock mode
    # ---------------------------------------------------------------

    if is_mock():
        return {
            "answer": _mock_answer(message, patient),
            "patient_id": patient_id,
            "provider": "mock",
        }

    # ---------------------------------------------------------------
    # Real LangChain agent
    # ---------------------------------------------------------------

    answer = _run_agent(
        patient_id,
        message,
    )

    return {
        "answer": answer,
        "patient_id": patient_id,
        "provider": "gemini",
    }
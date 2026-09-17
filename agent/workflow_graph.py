from __future__ import annotations

import uuid
import json
import re
import threading
from typing import Any, TypedDict

from agent.mcp_tools.client import call_mcp_tool_sync

try:
    from tracing import log_event
except ImportError:
    from agent.tracing import log_event

from langgraph.graph import END, START, StateGraph

from agent.document_check import (
    CHECKLIST_DOCUMENT_KEYS,
    CHECKLIST_DOCUMENT_LABELS,
)

from agent.llm_config import (
    get_chat_model,
    get_provider,
    is_mock,
    invoke_llm_traced,
)

try:
    from eval_judge import evaluate_workflow_decision
except ImportError:
    from agent.eval_judge import evaluate_workflow_decision


class PriorAuthState(TypedDict, total=False):
    conversation_id: str
    workflow_id: str
    _obs_nodes: list[str]

    patient_id: str
    document_status: dict

    patient: dict | None
    patient_found: bool

    insurance: dict

    available_facts: list

    documents: dict
    clinical_evidence: dict
    requirements: dict

    decision: str
    explanation: str
    next_action: str

    status: str
    error: str | None

    # -----------------------------------------------------------------------
    # FINAL WORKFLOW OBSERVABILITY
    # -----------------------------------------------------------------------
    workflow_outcome: str


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


# ---------------------------------------------------------------------------
# OBSERVABILITY NODE WRAPPER
# ---------------------------------------------------------------------------

def _traced_node(name: str, expected_prev: str):
    """Wrap a LangGraph node with transition/loop/guardrail telemetry."""

    def decorator(fn):

        def wrapped(state: PriorAuthState) -> dict:

            history = list(
                state.get("_obs_nodes", [])
            )

            previous = (
                history[-1]
                if history
                else "START"
            )

            loop = name in history
            invalid = previous != expected_prev

            try:

                result = fn(state) or {}

            except Exception as exc:

                try:

                    log_event(
                        conversation_id=(
                            state.get("conversation_id")
                            or "untagged"
                        ),
                        workflow_id=(
                            state.get("workflow_id")
                            or "untagged"
                        ),
                        step_name=name,
                        event_type="node_transition",
                        error=str(exc),
                        payload={
                            "from_node": previous,
                            "to_node": name,
                            "expected_prev": expected_prev,
                            "invalid_state_transition": invalid,
                            "agent_loop": loop,
                        },
                    )

                except Exception:
                    pass

                raise

            # ----------------------------------------------------------------
            # IMPORTANT:
            #
            # DO NOT mark completion/escalation here.
            #
            # Terminal workflow outcome is decided only by
            # finalize_workflow().
            # ----------------------------------------------------------------

            payload = {
                "from_node": previous,
                "to_node": name,
                "expected_prev": expected_prev,
                "invalid_state_transition": invalid,
                "agent_loop": loop,
            }

            explanation = str(
                result.get("explanation")
                or ""
            )

            next_action = str(
                result.get("next_action")
                or ""
            )

            guardrail_text = (
                f"{explanation} {next_action}"
                .lower()
            )

            guardrail_hit = bool(
                re.search(
                    r"\b(?:approved|denied|approve coverage|deny coverage)\b",
                    guardrail_text,
                )
            )

            if (
                guardrail_hit
                and "do not" not in guardrail_text
            ):

                try:

                    log_event(
                        conversation_id=(
                            state.get("conversation_id")
                            or "untagged"
                        ),
                        workflow_id=(
                            state.get("workflow_id")
                            or "untagged"
                        ),
                        step_name=name,
                        event_type="guardrail_violation",
                        payload={
                            "reason": (
                                "output_contains_approval_or_denial_language"
                            )
                        },
                    )

                except Exception:
                    pass

            try:

                log_event(
                    conversation_id=(
                        state.get("conversation_id")
                        or "untagged"
                    ),
                    workflow_id=(
                        state.get("workflow_id")
                        or "untagged"
                    ),
                    step_name=name,
                    event_type="node_transition",
                    payload=payload,
                )

            except Exception:
                pass

            return {
                **result,
                "_obs_nodes": history + [name],
            }

        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__

        return wrapped

    return decorator


# ---------------------------------------------------------------------------
# MCP RESPONSE PARSER
# ---------------------------------------------------------------------------

def parse_mcp_result(result) -> dict:
    """Extract the JSON object returned by an MCP tool."""

    if getattr(result, "is_error", False):

        raise RuntimeError(
            f"MCP tool error: {result}"
        )

    content = getattr(
        result,
        "content",
        [],
    )

    if not content:

        raise RuntimeError(
            "MCP tool returned no content"
        )

    text = getattr(
        content[0],
        "text",
        None,
    )

    if not text:

        raise RuntimeError(
            "MCP tool returned no text content"
        )

    try:

        return json.loads(text)

    except json.JSONDecodeError as exc:

        raise RuntimeError(
            f"MCP tool returned invalid JSON: {text}"
        ) from exc


# ---------------------------------------------------------------------------
# FACT EXTRACTION
# ---------------------------------------------------------------------------

def _facts_from_state(
    state: PriorAuthState,
) -> list[dict]:
    """Literal fields already loaded — never synthesized."""

    facts: list[dict] = []

    patient = (
        state.get("patient")
        or {}
    )

    for field in (
        "patient_id",
        "insurance_id",
        "diagnosis_code",
        "requested_procedure_code",
        "last_request_date",
        "name",
        "dob",
        "gender",
    ):

        if (
            field in patient
            and patient[field] not in (None, "")
        ):

            facts.append(
                {
                    "field": field,
                    "value": str(
                        patient[field]
                    ),
                }
            )

    insurance = (
        state.get("insurance")
        or {}
    )

    if insurance.get("rule_found"):

        facts.append(
            {
                "field": "requires_prior_auth",
                "value": str(
                    insurance.get(
                        "requires_prior_auth"
                    )
                ),
            }
        )

        if "historical_denial_rate" in insurance:

            facts.append(
                {
                    "field": "historical_denial_rate",
                    "value": str(
                        insurance.get(
                            "historical_denial_rate"
                        )
                    ),
                }
            )

    documents = (
        state.get("documents")
        or {}
    )

    if documents.get("present"):

        facts.append(
            {
                "field": "documents_present",
                "value": documents["present"],
            }
        )

    if documents.get("missing"):

        facts.append(
            {
                "field": "documents_missing",
                "value": documents["missing"],
            }
        )

    history = (
        state.get("patient") or {}
    ).get("_history_notes")

    if history:

        facts.append(
            {
                "field": "authorization_history_notes",
                "value": history,
            }
        )

    return facts


# ---------------------------------------------------------------------------
# NODE 1 — LOAD REQUEST
# ---------------------------------------------------------------------------

def load_request(
    state: PriorAuthState,
) -> dict:

    patient_id = (
        state.get("patient_id")
        or ""
    ).strip()

    try:

        patient = parse_mcp_result(
            call_mcp_tool_sync(
                "get_patient",
                {
                    "patient_id": patient_id,
                },
                conversation_id=(
                    state.get("conversation_id")
                ),
                workflow_id=(
                    state.get("workflow_id")
                ),
                expected_tool="get_patient",
            )
        )

        if patient.get(
            "patient_found"
        ) is False:

            found = False

            error = patient.get(
                "error",
                f"Patient '{patient_id}' not found",
            )

            patient = None

        else:

            found = True
            error = None

    except Exception as exc:

        patient = None
        found = False
        error = str(exc)

    insurance: dict = {
        "rule_found": False,
        "required_documents": [],
        "requires_prior_auth": None,
    }

    history_notes: list[str] = []

    if found and patient:

        try:

            # ---------------------------------------------------------------
            # Get insurance requirements through MCP
            # ---------------------------------------------------------------

            insurance = parse_mcp_result(
                call_mcp_tool_sync(
                    "get_insurance_requirements",
                    {
                        "insurance_id": patient.get(
                            "insurance_id",
                            "",
                        ),
                        "procedure_code": patient.get(
                            "requested_procedure_code",
                            "",
                        ),
                    },
                    conversation_id=(
                        state.get("conversation_id")
                    ),
                    workflow_id=(
                        state.get("workflow_id")
                    ),
                    expected_tool=(
                        "get_insurance_requirements"
                    ),
                )
            )

            # ---------------------------------------------------------------
            # Get authorization history through MCP
            # ---------------------------------------------------------------

            history_result = parse_mcp_result(
                call_mcp_tool_sync(
                    "get_authorization_history",
                    {
                        "patient_id": patient_id,
                    },
                    conversation_id=(
                        state.get("conversation_id")
                    ),
                    workflow_id=(
                        state.get("workflow_id")
                    ),
                    expected_tool=(
                        "get_authorization_history"
                    ),
                )
            )

            for record in history_result.get(
                "history",
                [],
            ):

                note = record.get(
                    "notes"
                )

                if note:

                    history_notes.append(
                        str(note)
                    )

            if history_notes:

                patient = {
                    **patient,
                    "_history_notes": history_notes,
                }

        except Exception as exc:

            error = str(exc)

    return {
        "patient": (
            _json_safe(patient)
            if patient
            else None
        ),
        "patient_found": found,
        "insurance": _json_safe(
            insurance
        ),
        "error": error,
        "status": (
            "REQUEST_INVALID"
            if not found
            else "LOADED"
        ),
    }


# ---------------------------------------------------------------------------
# NODE 2 — CHECK DOCUMENTS
# ---------------------------------------------------------------------------

def check_documents(
    state: PriorAuthState,
) -> dict:

    if not state.get(
        "patient_found"
    ):

        return {
            "documents": {
                "received": [],
                "missing": list(
                    CHECKLIST_DOCUMENT_KEYS
                ),
                "complete": False,
                "present": [],
                "required_documents": list(
                    CHECKLIST_DOCUMENT_KEYS
                ),
                "received_count": 0,
                "required_count": len(
                    CHECKLIST_DOCUMENT_KEYS
                ),
                "labels": {
                    key: CHECKLIST_DOCUMENT_LABELS.get(
                        key,
                        key,
                    )
                    for key in CHECKLIST_DOCUMENT_KEYS
                },
            },
            "status": "INCOMPLETE_DOCUMENTATION",
        }

    required = list(
        CHECKLIST_DOCUMENT_KEYS
    )

    # -----------------------------------------------------------------------
    # Call document checker through MCP
    # -----------------------------------------------------------------------

    result = parse_mcp_result(
        call_mcp_tool_sync(
            "check_document_status",
            {
                "document_status": (
                    state.get(
                        "document_status"
                    )
                    or {}
                ),
                "required_documents": required,
            },
            conversation_id=(
                state.get("conversation_id")
            ),
            workflow_id=(
                state.get("workflow_id")
            ),
            expected_tool="check_document_status",
        )
    )

    documents = {
        "received": result.get(
            "present",
            [],
        ),
        "missing": result.get(
            "missing",
            [],
        ),
        "complete": result.get(
            "complete",
            False,
        ),
        "present": result.get(
            "present",
            [],
        ),
        "required_documents": result.get(
            "required_documents",
            required,
        ),
        "received_count": result.get(
            "received_count",
            len(
                result.get(
                    "present",
                    [],
                )
            ),
        ),
        "required_count": result.get(
            "required_count",
            len(required),
        ),
        "labels": {
            key: CHECKLIST_DOCUMENT_LABELS.get(
                key,
                key,
            )
            for key in CHECKLIST_DOCUMENT_KEYS
        },
    }

    status = (
        "INCOMPLETE_DOCUMENTATION"
        if not documents["complete"]
        else "DOCUMENTS_COMPLETE"
    )

    return {
        "documents": documents,
        "status": status,
    }


# ---------------------------------------------------------------------------
# TEXT EXTRACTION
# ---------------------------------------------------------------------------

def _extract_text_content(
    content: Any,
) -> str:
    """Extract plain text from string, dict, or LangChain content block list."""

    if isinstance(
        content,
        str,
    ):

        return content

    if isinstance(
        content,
        list,
    ):

        parts = []

        for item in content:

            if isinstance(
                item,
                dict,
            ):

                parts.append(
                    str(
                        item.get(
                            "text",
                            "",
                        )
                    )
                )

            elif hasattr(
                item,
                "text",
            ):

                parts.append(
                    str(
                        getattr(
                            item,
                            "text",
                            "",
                        )
                    )
                )

            else:

                parts.append(
                    str(item)
                )

        return "".join(
            parts
        ).strip()

    return str(
        content or ""
    ).strip()


# ---------------------------------------------------------------------------
# NODE 3 — CLINICAL EVIDENCE
# ---------------------------------------------------------------------------

def analyze_clinical_evidence(
    state: PriorAuthState,
) -> dict:

    if not state.get(
        "patient_found"
    ):

        return {
            "available_facts": [],
            "clinical_evidence": {
                "source": "unavailable",
                "conditions": [],
                "symptoms": [],
                "findings": [],
                "treatments": [],
                "procedures": [],
                "evidence": [],
                "summary": (
                    "No patient record loaded."
                ),
                "facts": [],
            },
        }

    facts = _facts_from_state(
        state
    )

    patient = (
        state.get("patient")
        or {}
    )

    provider = get_provider()

    # -----------------------------------------------------------------------
    # MOCK LLM
    # -----------------------------------------------------------------------

    if is_mock():

        return {
            "available_facts": facts,
            "clinical_evidence": (
                _mock_structured_evidence(
                    facts,
                    patient,
                )
            ),
        }

    # -----------------------------------------------------------------------
    # LIVE GEMINI
    # -----------------------------------------------------------------------

    llm = get_chat_model()

    payload = {
        "instruction": (
            "You are a clinical evidence extraction component "
            "for a healthcare prior-authorization system. "
            "Analyze the following patient and authorization facts "
            "strictly and extract structured clinical evidence. "
            "Return a valid JSON object with EXACTLY these keys: "
            "{\"conditions\": list[str], "
            "\"symptoms\": list[str], "
            "\"findings\": list[str], "
            "\"treatments\": list[str], "
            "\"procedures\": list[str], "
            "\"evidence\": list[str], "
            "\"summary\": str}. "
            "CRITICAL: Never invent, extrapolate, or fabricate "
            "any diagnoses, symptoms, vitals, lab values, or medications. "
            "If a category is not present in the facts, return an empty list []."
        ),
        "patient_id": patient.get(
            "patient_id"
        ),
        "diagnosis_code": patient.get(
            "diagnosis_code"
        ),
        "requested_procedure_code": patient.get(
            "requested_procedure_code"
        ),
        "history_notes": patient.get(
            "_history_notes",
            [],
        ),
        "facts": facts,
    }

    prompt = (
        "Extract structured clinical evidence "
        "from these ground-truth prior-auth facts.\n"
        "Return ONLY a JSON object.\n"
        + json.dumps(
            payload,
            indent=2,
        )
    )

    response = invoke_llm_traced(
        llm,
        prompt,
        conversation_id=(
            state.get("conversation_id")
        ),
        workflow_id=(
            state.get("workflow_id")
        ),
        step_name=(
            "analyze_clinical_evidence"
        ),
    )

    raw_text = _extract_text_content(
        getattr(
            response,
            "content",
            response,
        )
    )

    parsed = (
        _parse_json_object(
            raw_text
        )
        or {}
    )

    clinical_evidence = {
        "source": provider,
        "conditions": (
            parsed.get(
                "conditions"
            )
            or _default_conditions(
                patient
            )
        ),
        "symptoms": (
            parsed.get(
                "symptoms"
            )
            or []
        ),
        "findings": (
            parsed.get(
                "findings"
            )
            or patient.get(
                "_history_notes",
                [],
            )
        ),
        "treatments": (
            parsed.get(
                "treatments"
            )
            or []
        ),
        "procedures": (
            parsed.get(
                "procedures"
            )
            or _default_procedures(
                patient
            )
        ),
        "evidence": (
            parsed.get(
                "evidence"
            )
            or [
                f"{f['field']}: {f['value']}"
                for f in facts
                if f.get("field")
                not in (
                    "documents_present",
                    "documents_missing",
                )
            ]
        ),
        "summary": (
            parsed.get(
                "summary"
            )
            or (
                f"Patient {patient.get('patient_id')} "
                f"authorization request for "
                f"{patient.get('requested_procedure_code')}."
            )
        ),
        "facts": facts,
    }

    return {
        "available_facts": facts,
        "clinical_evidence": clinical_evidence,
    }


# ---------------------------------------------------------------------------
# CLINICAL EVIDENCE HELPERS
# ---------------------------------------------------------------------------

def _default_conditions(
    patient: dict,
) -> list[str]:

    diag = patient.get(
        "diagnosis_code"
    )

    return (
        [f"Diagnosis Code: {diag}"]
        if diag
        else []
    )


def _default_procedures(
    patient: dict,
) -> list[str]:

    proc = patient.get(
        "requested_procedure_code"
    )

    return (
        [f"Requested Procedure: {proc}"]
        if proc
        else []
    )


def _mock_structured_evidence(
    facts: list[dict],
    patient: dict,
) -> dict:

    if not facts or not patient:

        return {
            "source": "unavailable",
            "conditions": [],
            "symptoms": [],
            "findings": [],
            "treatments": [],
            "procedures": [],
            "evidence": [],
            "summary": (
                "No clinical facts available."
            ),
            "facts": [],
        }

    conditions = _default_conditions(
        patient
    )

    procedures = _default_procedures(
        patient
    )

    findings = [
        str(h)
        for h in patient.get(
            "_history_notes",
            [],
        )
    ]

    return {
        "source": "mock",
        "conditions": conditions,
        "symptoms": [],
        "findings": findings,
        "treatments": [],
        "procedures": procedures,
        "evidence": [
            f"{item['field']}: {item['value']}"
            for item in facts
            if item.get("field")
            not in (
                "documents_present",
                "documents_missing",
            )
        ],
        "summary": (
            f"Structured facts for patient "
            f"{patient.get('patient_id')} "
            f"under diagnosis "
            f"{patient.get('diagnosis_code', 'N/A')}."
        ),
        "facts": facts,
    }


def _parse_json_object(
    text: str,
) -> dict | None:

    if not text:

        return None

    cleaned = text.strip()

    if cleaned.startswith("```"):

        cleaned = re.sub(
            r"^```(?:json)?\s*",
            "",
            cleaned,
        )

        cleaned = re.sub(
            r"\s*```$",
            "",
            cleaned,
        )

        cleaned = cleaned.strip()

    try:

        value = json.loads(
            cleaned
        )

        return (
            value
            if isinstance(value, dict)
            else None
        )

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            cleaned,
            re.DOTALL,
        )

        if not match:

            return None

        try:

            value = json.loads(
                match.group(0)
            )

            return (
                value
                if isinstance(value, dict)
                else None
            )

        except json.JSONDecodeError:

            return None


# ---------------------------------------------------------------------------
# NODE 4 — EVALUATE REQUIREMENTS
# ---------------------------------------------------------------------------

def evaluate_requirements(
    state: PriorAuthState,
) -> dict:

    documents = (
        state.get("documents")
        or {}
    )

    insurance = (
        state.get("insurance")
        or {}
    )

    patient_found = bool(
        state.get("patient_found")
    )

    docs_complete = bool(
        documents.get("complete")
    )

    rule_found = bool(
        insurance.get("rule_found")
    )

    requires_prior_auth = (
        insurance.get(
            "requires_prior_auth"
        )
    )

    missing_docs = (
        documents.get("missing")
        or []
    )

    checklist = [
        {
            "id": "req_patient_verified",
            "title": "Patient & Coverage Record",
            "description": (
                f"Verified patient record "
                f"{state.get('patient_id')} "
                f"with insurer "
                f"{insurance.get('insurance_id') or 'N/A'}"
            ),
            "satisfied": patient_found,
        },
        {
            "id": "req_policy_rule",
            "title": "Payer Prior Auth Policy",
            "description": (
                (
                    "Active policy on file "
                    "(Prior Auth required: "
                    f"{'Yes' if requires_prior_auth else 'No'})"
                )
                if rule_found
                else (
                    "No matching policy rule "
                    "found for payer + procedure"
                )
            ),
            "satisfied": rule_found,
        },
        {
            "id": "req_docs_complete",
            "title": "Documentation Completeness (6/6 Required)",
            "description": (
                "All 6 required documents on file"
                if docs_complete
                else (
                    f"Missing {len(missing_docs)} "
                    f"required document(s): "
                    f"{', '.join([CHECKLIST_DOCUMENT_LABELS.get(k, k) for k in missing_docs])}"
                )
            ),
            "satisfied": docs_complete,
        },
    ]

    satisfied = (
        patient_found
        and docs_complete
        and rule_found
    )

    if not patient_found:

        status = "REQUEST_INVALID"
        decision = "REQUEST_INVALID"

    elif not docs_complete:

        status = "INCOMPLETE_DOCUMENTATION"
        decision = "INCOMPLETE_DOCUMENTATION"

    elif not rule_found:

        status = "NEEDS_MANUAL_VERIFICATION"
        decision = "NEEDS_MANUAL_VERIFICATION"

    else:

        status = "READY_FOR_REVIEW"
        decision = "READY_FOR_REVIEW"

    requirements = {
        "rule_found": rule_found,
        "requires_prior_auth": (
            requires_prior_auth
        ),
        "required_documents": list(
            CHECKLIST_DOCUMENT_KEYS
        ),
        "insurance_required_documents": (
            insurance.get(
                "required_documents"
            )
            or []
        ),
        "documents_complete": (
            docs_complete
        ),
        "satisfied": satisfied,
        "checklist": checklist,
        "approval_not_automated": True,
    }

    return {
        "requirements": requirements,
        "status": status,
        "decision": decision,
    }


# ---------------------------------------------------------------------------
# NODE 5 — GENERATE DECISION
# ---------------------------------------------------------------------------

def generate_decision(
    state: PriorAuthState,
) -> dict:

    decision = (
        state.get("decision")
        or state.get("status")
        or "UNKNOWN"
    )

    facts = (
        state.get("available_facts")
        or _facts_from_state(
            state
        )
    )

    documents = (
        state.get("documents")
        or {}
    )

    missing = (
        documents.get("missing")
        or []
    )

    present = (
        documents.get("received")
        or documents.get("present")
        or []
    )

    clinical_evidence = (
        state.get("clinical_evidence")
        or {}
    )

    mock_explanation, mock_next_action = (
        _deterministic_decision_text(
            decision,
            present,
            missing,
            facts,
        )
    )

    if (
        is_mock()
        or not state.get(
            "patient_found"
        )
    ):

        return {
            "decision": decision,
            "explanation": mock_explanation,
            "next_action": mock_next_action,
        }

    # -----------------------------------------------------------------------
    # LIVE GEMINI
    # -----------------------------------------------------------------------

    llm = get_chat_model()

    missing_labels = [
        CHECKLIST_DOCUMENT_LABELS.get(
            k,
            k,
        )
        for k in missing
    ]

    payload = {
        "instruction": (
            "You are a healthcare prior authorization "
            "review assistant. "
            "Write a concise AI-Generated Review Note "
            "and Next Action for a clinician reviewer. "
            "Use only the provided structured state. "
            "Do not approve or deny coverage. "
            "Return JSON: "
            "{\"explanation\": str, "
            "\"next_action\": str}."
        ),
        "decision_status": decision,
        "documents_received": [
            CHECKLIST_DOCUMENT_LABELS.get(
                k,
                k,
            )
            for k in present
        ],
        "documents_missing": missing_labels,
        "clinical_summary": clinical_evidence.get(
            "summary",
            "",
        ),
        "conditions": clinical_evidence.get(
            "conditions",
            [],
        ),
        "procedures": clinical_evidence.get(
            "procedures",
            [],
        ),
        "findings": clinical_evidence.get(
            "findings",
            [],
        ),
    }

    prompt = (
        "Generate a review note and next action "
        "for this prior auth request.\n"
        "Return ONLY a JSON object with "
        "'explanation' and 'next_action'.\n"
        + json.dumps(
            payload,
            indent=2,
        )
    )

    response = invoke_llm_traced(
        llm,
        prompt,
        conversation_id=(
            state.get("conversation_id")
        ),
        workflow_id=(
            state.get("workflow_id")
        ),
        step_name="generate_decision",
    )

    raw_text = _extract_text_content(
        getattr(
            response,
            "content",
            response,
        )
    )

    parsed = (
        _parse_json_object(
            raw_text
        )
        or {}
    )

    explanation = (
        parsed.get(
            "explanation"
        )
        or raw_text
        or mock_explanation
    )

    next_action = (
        parsed.get(
            "next_action"
        )
        or mock_next_action
    )

    # Score this LLM-written explanation/next_action against the
    # deterministic ground-truth decision computed above.
    #
    # This runs in a background thread, NOT inline, because it makes its
    # own LLM call to the judge model. Running it synchronously would add
    # a 3rd sequential LLM call to every workflow run, directly extending
    # the latency the user is waiting on for a score that only feeds the
    # observability dashboard -- nobody is waiting on it. The thread is
    # fire-and-forget; evaluate_workflow_decision already swallows its
    # own errors internally, so a judge failure here can never surface
    # anywhere except the [eval_judge] diagnostic prints.
    try:
        threading.Thread(
            target=evaluate_workflow_decision,
            kwargs=dict(
                conversation_id=state.get("conversation_id"),
                workflow_id=state.get("workflow_id"),
                system_prompt=payload["instruction"],
                source_facts=json.dumps(
                    {
                        k: v
                        for k, v in payload.items()
                        if k != "instruction"
                    },
                    default=str,
                ),
                deterministic_decision=decision,
                missing_documents=missing_labels,
                explanation=explanation,
                next_action=next_action,
            ),
            daemon=True,
        ).start()
    except Exception:
        pass

    return {
        "decision": decision,
        "explanation": explanation,
        "next_action": next_action,
    }


# ---------------------------------------------------------------------------
# DECISION TEXT
# ---------------------------------------------------------------------------

def _deterministic_decision_text(
    decision: str,
    present: list,
    missing: list,
    facts: list,
) -> tuple[str, str]:

    missing_labels = [
        CHECKLIST_DOCUMENT_LABELS.get(
            k,
            k,
        )
        for k in missing
    ]

    fact_lines = [
        f"{item.get('field')}={item.get('value')}"
        for item in facts
        if item.get("field")
        not in (
            "documents_present",
            "documents_missing",
        )
    ]

    fact_block = (
        "; ".join(
            fact_lines
        )
        if fact_lines
        else "none"
    )

    if (
        decision
        == "INCOMPLETE_DOCUMENTATION"
    ):

        explanation = (
            "Authorization cannot proceed — "
            "required documentation is incomplete. "
            f"Received {len(present)} / "
            f"{len(CHECKLIST_DOCUMENT_KEYS)} documents. "
            f"Missing: "
            f"{', '.join(missing_labels) if missing_labels else 'none'}. "
            f"Known record fields: {fact_block}. "
            "This is not an insurance approval or denial."
        )

        next_action = (
            "Submit the missing documentation "
            f"({', '.join(missing_labels)}) "
            "before proceeding to authorization review."
        )

        return (
            explanation,
            next_action,
        )

    if decision == "READY_FOR_REVIEW":

        explanation = (
            "Required documentation complete — "
            "ready for authorization review. "
            f"Received {len(present)} / "
            f"{len(CHECKLIST_DOCUMENT_KEYS)} documents. "
            f"Known record fields: {fact_block}. "
            "This workflow does not issue an insurance approval."
        )

        next_action = (
            "Proceed with clinical review "
            "and payer adjudication."
        )

        return (
            explanation,
            next_action,
        )

    if (
        decision
        == "NEEDS_MANUAL_VERIFICATION"
    ):

        explanation = (
            "Payer rule not found in standard guidelines. "
            f"Known record fields: {fact_block}."
        )

        next_action = (
            "Contact payer for manual "
            "authorization verification."
        )

        return (
            explanation,
            next_action,
        )

    explanation = (
        "The patient or request could not be loaded. "
        f"Known record fields: {fact_block}."
    )

    next_action = (
        "Verify patient record ID and re-submit."
    )

    return (
        explanation,
        next_action,
    )


def _deterministic_explanation(
    decision: str,
    present: list,
    missing: list,
    facts: list,
) -> str:

    explanation, _ = _deterministic_decision_text(
        decision,
        present,
        missing,
        facts,
    )

    return explanation


# ---------------------------------------------------------------------------
# NODE 6 — FINALIZE WORKFLOW
# ---------------------------------------------------------------------------

def finalize_workflow(
    state: PriorAuthState,
) -> dict:
    """
    Determine exactly one terminal workflow outcome.

    COMPLETED
        The workflow successfully produced its final review result.

    ESCALATED
        The request requires missing information or manual verification.

    FAILED
        The workflow could not successfully produce a usable result.
    """

    error = state.get("error")

    decision = (
        state.get("decision")
        or ""
    )

    status = (
        state.get("status")
        or ""
    )

    explanation = (
        state.get("explanation")
        or ""
    )

    next_action = (
        state.get("next_action")
        or ""
    )

    # -----------------------------------------------------------------------
    # FAILED
    # -----------------------------------------------------------------------

    if error:

        outcome = "FAILED"
        reason = "workflow_error"

    # -----------------------------------------------------------------------
    # ESCALATED
    #
    # These are not successful authorization-ready outcomes.
    # They require additional information or human intervention.
    # -----------------------------------------------------------------------

    elif decision in {
        "REQUEST_INVALID",
        "INCOMPLETE_DOCUMENTATION",
        "NEEDS_MANUAL_VERIFICATION",
    }:

        outcome = "ESCALATED"

        if decision == "REQUEST_INVALID":
            reason = "invalid_request"

        elif decision == "INCOMPLETE_DOCUMENTATION":
            reason = "missing_information"

        else:
            reason = "manual_verification_required"

    # -----------------------------------------------------------------------
    # COMPLETED
    # -----------------------------------------------------------------------

    elif (
        decision == "READY_FOR_REVIEW"
        and explanation
        and next_action
    ):

        outcome = "COMPLETED"
        reason = "final_review_result_generated"

    # -----------------------------------------------------------------------
    # FALLBACK FAILURE
    # -----------------------------------------------------------------------

    else:

        outcome = "FAILED"
        reason = "unexpected_terminal_state"

    payload = {
        "outcome": outcome,
        "reason": reason,
        "status": status,
        "decision": decision,
        "has_explanation": bool(
            explanation
        ),
        "has_next_action": bool(
            next_action
        ),
    }

    try:

        log_event(
            conversation_id=(
                state.get("conversation_id")
                or "untagged"
            ),
            workflow_id=(
                state.get("workflow_id")
                or "untagged"
            ),
            step_name="finalize_workflow",
            event_type="workflow_outcome",
            payload=payload,
        )

    except Exception:
        pass

    return {
        "workflow_outcome": outcome,
    }


# ---------------------------------------------------------------------------
# BUILD LANGGRAPH WORKFLOW
# ---------------------------------------------------------------------------

def build_workflow():

    graph = StateGraph(
        PriorAuthState
    )

    graph.add_node(
        "load_request",
        _traced_node(
            "load_request",
            "START",
        )(load_request),
    )

    graph.add_node(
        "check_documents",
        _traced_node(
            "check_documents",
            "load_request",
        )(check_documents),
    )

    graph.add_node(
        "analyze_clinical_evidence",
        _traced_node(
            "analyze_clinical_evidence",
            "check_documents",
        )(analyze_clinical_evidence),
    )

    graph.add_node(
        "evaluate_requirements",
        _traced_node(
            "evaluate_requirements",
            "analyze_clinical_evidence",
        )(evaluate_requirements),
    )

    graph.add_node(
        "generate_decision",
        _traced_node(
            "generate_decision",
            "evaluate_requirements",
        )(generate_decision),
    )

    # -----------------------------------------------------------------------
    # NEW TERMINAL NODE
    # -----------------------------------------------------------------------

    graph.add_node(
        "finalize_workflow",
        _traced_node(
            "finalize_workflow",
            "generate_decision",
        )(finalize_workflow),
    )

    graph.add_edge(
        START,
        "load_request",
    )

    graph.add_edge(
        "load_request",
        "check_documents",
    )

    graph.add_edge(
        "check_documents",
        "analyze_clinical_evidence",
    )

    graph.add_edge(
        "analyze_clinical_evidence",
        "evaluate_requirements",
    )

    graph.add_edge(
        "evaluate_requirements",
        "generate_decision",
    )

    # -----------------------------------------------------------------------
    # NEW TERMINAL FLOW
    # -----------------------------------------------------------------------

    graph.add_edge(
        "generate_decision",
        "finalize_workflow",
    )

    graph.add_edge(
        "finalize_workflow",
        END,
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# WORKFLOW SINGLETON
# ---------------------------------------------------------------------------

_APP = None


def get_workflow():

    global _APP

    if _APP is None:

        _APP = build_workflow()

    return _APP


# ---------------------------------------------------------------------------
# PUBLIC WORKFLOW API
# ---------------------------------------------------------------------------

def run_workflow(
    patient_id: str,
    document_status: dict,
    conversation_id: str | None = None,
) -> dict:

    workflow_id = str(
        uuid.uuid4()
    )

    conversation_id = (
        conversation_id
        or workflow_id
    )

    app = get_workflow()

    result = app.invoke(
        {
            "conversation_id": conversation_id,
            "workflow_id": workflow_id,
            "patient_id": patient_id,
            "document_status": (
                document_status
                or {}
            ),
        }
    )

    return {
        "conversation_id": conversation_id,
        "workflow_id": workflow_id,

        # -------------------------------------------------------------------
        # EXISTING WORKFLOW STATUS
        # -------------------------------------------------------------------

        "status": result.get(
            "status"
        ),

        "decision": result.get(
            "decision"
        ),

        # -------------------------------------------------------------------
        # NEW OBSERVABILITY OUTCOME
        # -------------------------------------------------------------------

        "workflow_outcome": result.get(
            "workflow_outcome"
        ),

        "explanation": result.get(
            "explanation"
        ),

        "next_action": result.get(
            "next_action"
        ),

        "documents": (
            result.get(
                "documents"
            )
            or {}
        ),

        "clinical_evidence": (
            result.get(
                "clinical_evidence"
            )
            or {}
        ),

        "requirements": (
            result.get(
                "requirements"
            )
            or {}
        ),

        "patient": result.get(
            "patient"
        ),

        "insurance": result.get(
            "insurance"
        ),

        "error": result.get(
            "error"
        ),

        "llm_provider": get_provider(),
    }
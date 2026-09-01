"""MVP tests for checklist checking, LangGraph workflow, and Flask API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "backend"))

os.environ["LLM_PROVIDER"] = "mock"

from document_check import CHECKLIST_DOCUMENT_KEYS, DocumentCheckModule, check_from_status
from llm_config import LLMConfigError, create_gemini_chat_model, get_provider, is_mock
from workflow_graph import build_workflow, run_workflow


INCOMPLETE_STATUS = {
    "clinical_note": True,
    "diagnosis_evidence": True,
    "lab_report": True,
    "imaging_report": False,
    "medication_history": True,
    "physician_order": False,
}

COMPLETE_STATUS = {key: True for key in CHECKLIST_DOCUMENT_KEYS}


def test_case_a_missing_documents_checker():
    result = check_from_status(INCOMPLETE_STATUS, CHECKLIST_DOCUMENT_KEYS)
    assert result["received_count"] == 4
    assert result["required_count"] == 6
    assert result["complete"] is False
    assert result["missing"] == ["imaging_report", "physician_order"]


def test_case_a_missing_documents_workflow():
    result = run_workflow("P001", INCOMPLETE_STATUS)
    assert result["status"] == "INCOMPLETE_DOCUMENTATION"
    assert result["decision"] == "INCOMPLETE_DOCUMENTATION"
    assert result["documents"]["missing"] == ["imaging_report", "physician_order"]
    assert result["documents"]["complete"] is False
    assert result["clinical_evidence"]["source"] == "mock"
    assert isinstance(result["clinical_evidence"]["facts"], list)


def test_case_b_all_documents_present():
    result = run_workflow("P001", COMPLETE_STATUS)
    assert result["status"] == "READY_FOR_REVIEW"
    assert result["decision"] == "READY_FOR_REVIEW"
    assert result["documents"]["complete"] is True
    assert result["documents"]["missing"] == []
    assert result["documents"]["received_count"] == 6


def test_case_d_mock_llm_no_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    graph = build_workflow()
    state = graph.invoke({"patient_id": "P001", "document_status": COMPLETE_STATUS})
    assert state["clinical_evidence"]["source"] == "mock"
    for fact in state["clinical_evidence"]["facts"]:
        assert "field" in fact
        assert "value" in fact
    assert "G47.33" in str(state["clinical_evidence"]["facts"])
    assert "invented" not in (state.get("explanation") or "").lower()


def test_mock_does_not_invent_when_no_facts_beyond_record():
    result = run_workflow("P001", INCOMPLETE_STATUS)
    fact_blob = str(result["clinical_evidence"]["facts"]).lower()
    assert "sleep study apnea index" not in fact_blob
    assert "made-up" not in fact_blob


def test_existing_file_based_check_still_works():
    module = DocumentCheckModule()
    result = module.check("PAT-1124", ["clinical_notes", "physician_referral"])
    assert result["all_documents_present"] is True
    missing = module.check("PAT-1124", ["clinical_notes", "imaging_report"])
    assert "imaging_report" in missing["missing"]
    assert missing["all_documents_present"] is False


def test_case_e_gemini_configuration_without_network(monkeypatch):
    captured = {}

    class FakeChat:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-live")
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    monkeypatch.setitem(sys.modules, "langchain_google_genai", MagicMock())

    with patch("langchain_google_genai.ChatGoogleGenerativeAI", FakeChat):
        model = create_gemini_chat_model()

    assert isinstance(model, FakeChat)
    assert captured.get("google_api_key") == "test-key-not-live"
    assert captured.get("model") == "gemini-2.0-flash"


def test_gemini_requires_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(LLMConfigError):
        create_gemini_chat_model(api_key="")


def test_graph_nodes_are_connected():
    graph = build_workflow()
    names = set(graph.get_graph().nodes)
    for node in (
        "load_request",
        "check_documents",
        "analyze_clinical_evidence",
        "evaluate_requirements",
        "generate_decision",
    ):
        assert node in names


def test_all_five_nodes_execute_and_pass_state():
    """Verify that every single LangGraph node actually executes and enriches state."""
    graph = build_workflow()
    state = graph.invoke({"patient_id": "P001", "document_status": COMPLETE_STATUS})

    # 1. load_request verification
    assert state.get("patient_found") is True
    assert isinstance(state.get("patient"), dict)
    assert state["patient"]["patient_id"] == "P001"
    assert "insurance" in state
    assert state["insurance"].get("rule_found") is True

    # 2. check_documents verification
    assert "documents" in state
    assert state["documents"]["complete"] is True
    assert state["documents"]["received_count"] == 6
    assert state["documents"]["required_count"] == 6
    assert state["documents"]["missing"] == []
    assert len(state["documents"]["received"]) == 6

    # 3. analyze_clinical_evidence verification
    assert "clinical_evidence" in state
    assert "available_facts" in state
    assert isinstance(state["clinical_evidence"]["facts"], list)
    assert len(state["clinical_evidence"]["facts"]) > 0

    # 4. evaluate_requirements verification
    assert "requirements" in state
    assert state["requirements"]["satisfied"] is True
    assert state["requirements"]["rule_found"] is True
    assert state["requirements"]["documents_complete"] is True

    # 5. generate_decision verification
    assert state.get("decision") == "READY_FOR_REVIEW"
    assert isinstance(state.get("explanation"), str)
    assert len(state["explanation"]) > 0
    assert "ready for authorization review" in state["explanation"].lower()


def test_unknown_patient_graph_execution_no_fake_facts():
    """Verify that graph handles unknown patient without inventing facts."""
    graph = build_workflow()
    state = graph.invoke({"patient_id": "NON_EXISTENT_ID", "document_status": {}})
    assert state["patient_found"] is False
    assert state["patient"] is None
    assert state["clinical_evidence"]["facts"] == []
    assert state["decision"] == "REQUEST_INVALID"
    assert "could not be loaded" in state["explanation"]


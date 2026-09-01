"""HTTP tests for POST /api/run-workflow."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))
sys.path.insert(0, str(ROOT / "backend"))

os.environ["LLM_PROVIDER"] = "mock"

from app import app  # noqa: E402
from document_check import CHECKLIST_DOCUMENT_KEYS  # noqa: E402


INCOMPLETE_STATUS = {
    "clinical_note": True,
    "diagnosis_evidence": True,
    "lab_report": True,
    "imaging_report": False,
    "medication_history": True,
    "physician_order": False,
}


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_case_c_unknown_patient(client):
    response = client.post(
        "/api/run-workflow",
        json={"patient_id": "DOES-NOT-EXIST", "document_status": INCOMPLETE_STATUS},
    )
    assert response.status_code == 404
    body = response.get_json()
    assert "error" in body
    assert "DOES-NOT-EXIST" in body["error"]


def test_case_c_missing_patient_id(client):
    response = client.post("/api/run-workflow", json={"document_status": {}})
    assert response.status_code == 400
    assert "patient_id" in response.get_json()["error"]


def test_case_c_invalid_document_status(client):
    response = client.post(
        "/api/run-workflow",
        json={"patient_id": "P001", "document_status": ["not", "an", "object"]},
    )
    assert response.status_code == 400


def test_run_workflow_incomplete(client):
    response = client.post(
        "/api/run-workflow",
        json={"patient_id": "P001", "document_status": INCOMPLETE_STATUS},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "INCOMPLETE_DOCUMENTATION"
    assert body["documents"]["missing"] == ["imaging_report", "physician_order"]
    assert "clinical_evidence" in body
    assert "requirements" in body
    assert "explanation" in body


def test_run_workflow_complete(client):
    response = client.post(
        "/api/run-workflow",
        json={
            "patient_id": "P001",
            "document_status": {key: True for key in CHECKLIST_DOCUMENT_KEYS},
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "READY_FOR_REVIEW"
    assert body["documents"]["complete"] is True


def test_missing_json_body(client):
    response = client.post(
        "/api/run-workflow",
        data="raw non-json text",
        content_type="text/plain",
    )
    assert response.status_code == 400
    body = response.get_json()
    assert "error" in body


def test_invalid_document_status_unknown_keys(client):
    response = client.post(
        "/api/run-workflow",
        json={
            "patient_id": "P001",
            "document_status": {"invalid_doc_type": True},
        },
    )
    assert response.status_code == 400
    body = response.get_json()
    assert "Unknown document_status keys" in body["error"]


def test_invalid_document_status_non_boolean_values(client):
    response = client.post(
        "/api/run-workflow",
        json={
            "patient_id": "P001",
            "document_status": {"clinical_note": "yes"},
        },
    )
    assert response.status_code == 400
    body = response.get_json()
    assert "must be a boolean" in body["error"]


def test_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert "llm_provider" in body


def test_index_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"PriorAuth" in response.data


def test_patients_list_endpoint(client):
    response = client.get("/api/patients")
    assert response.status_code == 200
    body = response.get_json()
    assert "patients" in body
    assert isinstance(body["patients"], list)
    assert len(body["patients"]) > 0
    patient_ids = [p["patient_id"] for p in body["patients"]]
    assert "P001" in patient_ids


def test_patient_detail_endpoint(client):
    response = client.get("/api/patient/P001")
    assert response.status_code == 200
    body = response.get_json()
    assert "patient" in body
    assert body["patient"]["patient_id"] == "P001"
    assert "insurance" in body
    assert body["insurance"]["requires_prior_auth"] is True



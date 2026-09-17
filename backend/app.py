from __future__ import annotations

import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agent"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from chat_agent import chat_with_patient
from document_check import CHECKLIST_DOCUMENT_KEYS
from insurance_requirements import InsuranceRequirementsModule
from patient_info import PatientInfoModule, PatientNotFoundError
from agent.workflow_graph import run_workflow

try:
    from tracing import log_event, read_events
    from metrics import compute_all_metrics
except ImportError:
    from agent.tracing import log_event, read_events
    from agent.metrics import compute_all_metrics

FRONTEND_DIR = ROOT / "frontend"

app = Flask(__name__)


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return response


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "llm_provider": os.getenv("LLM_PROVIDER", "mock"),
        "llm_model": os.getenv("LLM_MODEL", "gemini-3.6-flash"),
    })


def _client_error(message: str, status: int):
    return jsonify({"error": message, "status": "REQUEST_INVALID"}), status


@app.route("/api/patients", methods=["GET"])
def api_get_patients():
    """List all available patients from patients.csv."""
    try:
        module = PatientInfoModule()
        patients = module.patients.to_dict(orient="records")
        return jsonify({"patients": patients, "count": len(patients)})
    except Exception as exc:
        return jsonify({"error": str(exc), "status": "SERVER_ERROR"}), 500


@app.route("/api/patient/<patient_id>", methods=["GET"])
def api_get_patient(patient_id: str):
    """Return detailed patient info, past auth history, and payer requirements."""
    patient_id = (patient_id or "").strip()
    patients_mod = PatientInfoModule()
    insurers_mod = InsuranceRequirementsModule()

    try:
        patient = patients_mod.get_patient(patient_id)
    except PatientNotFoundError as exc:
        return _client_error(str(exc), 404)

    history = patients_mod.get_history(patient_id)
    insurance = insurers_mod.check(
        patient.get("insurance_id", ""),
        patient.get("requested_procedure_code", ""),
    )

    return jsonify({
        "patient": patient,
        "history": history,
        "history_count": len(history),
        "insurance": insurance,
    })


@app.route("/api/run-workflow", methods=["POST", "OPTIONS"])
def api_run_workflow():
    if request.method == "OPTIONS":
        return ("", 204)

    if not request.is_json:
        return _client_error("Request body must be JSON.", 400)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _client_error("Request body must be a JSON object.", 400)

    patient_id = payload.get("patient_id")
    if not patient_id or not isinstance(patient_id, str) or not patient_id.strip():
        return _client_error("patient_id is required.", 400)
    patient_id = patient_id.strip()

    document_status = payload.get("document_status", {})
    if not isinstance(document_status, dict):
        return _client_error("document_status must be an object of booleans.", 400)

    unknown_keys = set(document_status) - set(CHECKLIST_DOCUMENT_KEYS)
    if unknown_keys:
        return _client_error(
            "Unknown document_status keys: " + ", ".join(sorted(unknown_keys)),
            400,
        )

    for key, value in document_status.items():
        if not isinstance(value, bool):
            return _client_error(
                f"document_status.{key} must be a boolean.",
                400,
            )

    try:
        PatientInfoModule().get_patient(patient_id)
    except PatientNotFoundError as exc:
        return _client_error(str(exc), 404)

    try:
        result = run_workflow(patient_id, document_status)
    except Exception as exc:
        return jsonify(
            {
                "error": f"Workflow failed: {str(exc)}",
                "status": "WORKFLOW_ERROR",
            }
        ), 500

    return jsonify(
        {
            "status": result.get("status"),
            "decision": result.get("decision"),
            "explanation": result.get("explanation"),
            "next_action": result.get("next_action"),
            "documents": result.get("documents") or {},
            "clinical_evidence": result.get("clinical_evidence") or {},
            "requirements": result.get("requirements") or {},
            "patient": result.get("patient"),
            "insurance": result.get("insurance"),
            "llm_provider": result.get("llm_provider"),
            # so the frontend can tag a later /api/feedback call to this
            # exact run
            "conversation_id": result.get("conversation_id"),
            "workflow_id": result.get("workflow_id"),
        }
    )
@app.route("/api/chat", methods=["POST", "OPTIONS"])
def api_chat():
    if request.method == "OPTIONS":
        return ("", 204)

    if not request.is_json:
        return _client_error("Request body must be JSON.", 400)

    payload = request.get_json(silent=True)

    if not isinstance(payload, dict):
        return _client_error("Request body must be a JSON object.", 400)

    patient_id = payload.get("patient_id")
    message = payload.get("message")

    if not isinstance(patient_id, str) or not patient_id.strip():
        return _client_error("patient_id is required.", 400)

    if not isinstance(message, str) or not message.strip():
        return _client_error("message is required.", 400)

    try:
        result = chat_with_patient(
            patient_id.strip(),
            message.strip(),
        )

        if result.get("error"):
            return _client_error(result["error"], 404)

        return jsonify(result)

    except Exception as exc:
        return jsonify({
            "error": f"Chat failed: {str(exc)}",
            "status": "CHAT_ERROR",
        }), 500

@app.route("/api/feedback", methods=["POST", "OPTIONS"])
def api_feedback():
    if request.method == "OPTIONS":
        return ("", 204)

    if not request.is_json:
        return _client_error("Request body must be JSON.", 400)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return _client_error("Request body must be a JSON object.", 400)

    conversation_id = payload.get("conversation_id")
    workflow_id = payload.get("workflow_id")
    feedback = payload.get("feedback")

    if feedback not in ("up", "down"):
        return _client_error("feedback must be 'up' or 'down'.", 400)

    if not conversation_id or not workflow_id:
        return _client_error("conversation_id and workflow_id are required.", 400)

    log_event(
        conversation_id=conversation_id,
        workflow_id=workflow_id,
        step_name="user_feedback",
        event_type="user_feedback",
        user_feedback=feedback,
        payload={"comment": payload.get("comment")},
    )
    return jsonify({"status": "ok"})


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    """Serves the same numbers scripts/compute_metrics.py prints -- both
    read agent/metrics.py, so the dashboard and the CLI can never drift
    out of sync with each other."""
    events = list(read_events())
    if not events:
        return jsonify({"summary": {}, "trend": {"days": []}, "eval": {}, "generated_at": None})
    return jsonify(compute_all_metrics(events))


@app.route("/dashboard")
def dashboard():
    return send_from_directory(FRONTEND_DIR, "dashboard.html")


def main():
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=False)


if __name__ == "__main__":
    main()


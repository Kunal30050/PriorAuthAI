"""
Phase 4 — Document Check Module
---------------------------------
Given a patient_id + a list of required_documents (from Phase 3's insurance
check), this module checks documents.csv to see which of those required
documents are actually on file, and which are missing.

This module only reads documents.csv. It does not decide insurance rules
(Phase 3) or make the final authorize/reject call (Phase 5).
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DOCUMENTS_FILE = os.path.join(DATA_DIR, "documents.csv")

# MVP abstract checklist keys (frontend / POST /api/run-workflow).
# Distinct from file-based document_type values in documents.csv.
CHECKLIST_DOCUMENT_KEYS = [
    "clinical_note",
    "diagnosis_evidence",
    "lab_report",
    "imaging_report",
    "medication_history",
    "physician_order",
]

CHECKLIST_DOCUMENT_LABELS = {
    "clinical_note": "Clinical / Physician Note",
    "diagnosis_evidence": "Diagnosis Evidence",
    "lab_report": "Lab Report",
    "imaging_report": "Imaging / Radiology Report",
    "medication_history": "Medication History",
    "physician_order": "Physician Order / Prescription",
}


def check_from_status(document_status, required_documents) -> dict:
    """Compare abstract boolean checklist status against required document keys.

    Same present/missing semantics as DocumentCheckModule.check(), but the
    source of "submitted" is structured status rather than documents.csv.
    """
    status = document_status or {}
    if not isinstance(status, dict):
        raise TypeError("document_status must be an object/dict of booleans")

    required = list(required_documents or [])
    required_set = set(required)

    submitted_set = {key for key, present in status.items() if present is True}
    missing = sorted(required_set - submitted_set)
    present = sorted(required_set & submitted_set)

    return {
        "required_documents": sorted(required_set),
        "submitted_documents": sorted(submitted_set),
        "present": present,
        "missing": missing,
        "received": present,
        "all_documents_present": len(missing) == 0,
        "complete": len(missing) == 0,
        "received_count": len(present),
        "required_count": len(required_set),
    }


class DocumentCheckModule:
    def __init__(self):
        self.documents = pd.read_csv(DOCUMENTS_FILE, dtype=str)

    def get_submitted_documents(self, patient_id: str) -> list[str]:
        """Return the list of document_types on file (status=submitted) for
        this patient."""
        rows = self.documents[
            (self.documents["patient_id"] == patient_id)
            & (self.documents["status"] == "submitted")
        ]
        return rows["document_type"].tolist()

    def check(self, patient_id: str, required_documents: list[str]) -> dict:
        """Compare required_documents against what's actually on file."""
        submitted = self.get_submitted_documents(patient_id)
        submitted_set = set(submitted)
        required_set = set(required_documents)

        missing = sorted(required_set - submitted_set)
        present = sorted(required_set & submitted_set)

        return {
            "patient_id": patient_id,
            "required_documents": sorted(required_set),
            "submitted_documents": sorted(submitted_set),
            "present": present,
            "missing": missing,
            "all_documents_present": len(missing) == 0,
        }

    def check_from_status(self, document_status, required_documents) -> dict:
        """Checklist comparison; does not read documents.csv."""
        return check_from_status(document_status, required_documents)


# ---------- Quick manual test ----------
if __name__ == "__main__":
    module = DocumentCheckModule()

    print("=== Test 1: Pick a real patient and see what they've submitted ===")
    sample_patient = module.documents["patient_id"].iloc[0]
    submitted = module.get_submitted_documents(sample_patient)
    print(f"Patient {sample_patient} has submitted: {submitted}")

    print("\n=== Test 2: Check against a required list they fully satisfy ===")
    result = module.check(sample_patient, submitted[:2] if len(submitted) >= 2 else submitted)
    print(result)

    print("\n=== Test 3: Check against a required list with something missing ===")
    result = module.check(sample_patient, submitted + ["nonexistent_document_type"])
    print(result)

    print("\n=== Test 4: Patient with zero documents on file ===")
    all_patients = set(module.documents["patient_id"].unique())
    result = module.check("PAT-00000-NOT-REAL", ["physician_referral", "clinical_notes"])
    print(result)

"""
Phase 2 — Patient Info Module
------------------------------
Given a patient_id (+ optionally a requested procedure_code), this module:
  1. Looks up the patient's basic info from patients.csv
  2. Looks up the patient's past authorization history from
     patient_authorization_history.csv (the real uploaded dataset)
  3. Flags any past history specifically relevant to the requested procedure

This module only READS data — it does not check insurance rules or
documents. That's Phase 3 and Phase 4.
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PATIENTS_FILE = os.path.join(DATA_DIR, "patients.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "patient_authorization_history.csv")


class PatientNotFoundError(Exception):
    pass


class PatientInfoModule:
    def __init__(self):
        self.patients = pd.read_csv(PATIENTS_FILE, dtype=str)
        self.history = pd.read_csv(HISTORY_FILE, dtype=str)

    def get_patient(self, patient_id: str) -> dict:
        """Return the patient's basic record from patients.csv."""
        row = self.patients[self.patients["patient_id"] == patient_id]
        if row.empty:
            raise PatientNotFoundError(
                f"No patient found with ID '{patient_id}' in patients.csv"
            )
        return row.iloc[0].to_dict()

    def get_history(self, patient_id: str) -> list[dict]:
        """Return all past authorization requests for this patient_id,
        most recent first. Returns an empty list if none found — this is
        NOT an error, since a first-time patient simply has no history."""
        rows = self.history[self.history["patient_id"] == patient_id]
        rows = rows.sort_values("request_date", ascending=False)
        return rows.to_dict(orient="records")

    def get_relevant_history(self, patient_id: str, procedure_code: str) -> list[dict]:
        """Filter history to records matching the requested procedure/service
        code specifically — useful to check 'has this exact thing been
        requested before, and what happened?'"""
        full_history = self.get_history(patient_id)
        return [
            record for record in full_history
            if procedure_code.upper() in str(record.get("service_code", "")).upper()
        ]

    def summarize(self, patient_id: str, procedure_code: str = None) -> dict:
        """One-call summary combining patient info + history, used by the
        agent's main workflow in later phases."""
        summary = {"patient_id": patient_id}

        try:
            summary["patient_info"] = self.get_patient(patient_id)
            summary["patient_found"] = True
        except PatientNotFoundError:
            summary["patient_info"] = None
            summary["patient_found"] = False

        summary["history"] = self.get_history(patient_id)
        summary["history_count"] = len(summary["history"])

        if procedure_code:
            summary["relevant_history"] = self.get_relevant_history(patient_id, procedure_code)
        else:
            summary["relevant_history"] = []

        return summary


# ---------- Quick manual test ----------
if __name__ == "__main__":
    module = PatientInfoModule()

    print("=== Test 1: Mock patient (P001), from patients.csv ===")
    result = module.summarize("P001", "72148")
    print(f"Patient found: {result['patient_found']}")
    print(f"Patient info: {result['patient_info']}")
    print(f"History records: {result['history_count']} (expected 0 — mock patients aren't in the real history file)")

    print("\n=== Test 2: Real patient from uploaded history (PAT-19873) ===")
    result = module.summarize("PAT-19873", "CPT72148")
    print(f"Patient found in patients.csv: {result['patient_found']}")
    print(f"Total history records: {result['history_count']}")
    print(f"Relevant history for CPT72148: {len(result['relevant_history'])}")
    for record in result["history"]:
        print(f"  - {record['request_date']}: {record['service_description']} "
              f"({record['service_code']}) -> {record['approval_status']}")

    print("\n=== Test 3: Unknown patient ===")
    result = module.summarize("UNKNOWN123")
    print(f"Patient found: {result['patient_found']}")
    print(f"History records: {result['history_count']}")

"""
Phase 3 — Insurance Requirements Module
-----------------------------------------
Given an insurance_id (payer_id) + procedure_code (service_code), this module
checks insurance_rules.csv to determine:
  - whether prior authorization is required at all
  - which documents are required if so
  - the historical denial rate for that payer+procedure combo (context only)

This module only reads insurance_rules.csv. It does not touch patient data
or documents — that's patient_info.py (Phase 2) and documents.py (Phase 4).
"""

import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RULES_FILE = os.path.join(DATA_DIR, "insurance_rules.csv")


class InsuranceRequirementsModule:
    def __init__(self):
        self.rules = pd.read_csv(RULES_FILE, dtype=str)

    def check(self, insurance_id: str, procedure_code: str) -> dict:
        """Look up the rule for this payer + procedure combo."""
        row = self.rules[
            (self.rules["payer_id"] == insurance_id)
            & (self.rules["service_code"] == procedure_code)
        ]

        if row.empty:
            return {
                "insurance_id": insurance_id,
                "procedure_code": procedure_code,
                "rule_found": False,
                "requires_prior_auth": None,
                "required_documents": [],
                "message": (
                    f"No rule on file for payer '{insurance_id}' + procedure "
                    f"'{procedure_code}'. Cannot confirm prior auth requirements — "
                    f"treat as requiring manual payer verification."
                ),
            }

        record = row.iloc[0]
        requires_auth = record["requires_prior_auth"].strip().lower() == "yes"
        docs = record["required_documents"]
        required_documents = [d for d in docs.split("|") if d] if isinstance(docs, str) else []

        return {
            "insurance_id": insurance_id,
            "procedure_code": procedure_code,
            "rule_found": True,
            "requires_prior_auth": requires_auth,
            "required_documents": required_documents,
            "historical_denial_rate": float(record.get("historical_denial_rate", 0) or 0),
            "based_on_n_records": int(record.get("based_on_n_records", 0) or 0),
        }


# ---------- Quick manual test ----------
if __name__ == "__main__":
    module = InsuranceRequirementsModule()

    print("=== Test 1: Known payer+procedure combo ===")
    result = module.check("PAY-0445", "HCPCS-E0607")
    print(result)

    print("\n=== Test 2: Known combo with no prior auth needed (if one exists) ===")
    sample = module.rules[module.rules["requires_prior_auth"] == "no"]
    if not sample.empty:
        r = sample.iloc[0]
        result = module.check(r["payer_id"], r["service_code"])
        print(result)
    else:
        print("(No 'no prior auth needed' combos found in current data)")

    print("\n=== Test 3: Unknown payer+procedure combo ===")
    result = module.check("PAY-9999", "CPT00000")
    print(result)

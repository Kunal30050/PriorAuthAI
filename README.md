# Prior Authorization Agent — Build Plan (Waterfall)

Building this in locked phases. Each phase is finished and reviewed before
the next one starts.

1. Requirements & Data Design
2. Patient Info Module
3. Insurance Requirements Module
4. Document Check Module
5. Decision Logic (missing-info vs. draft-generation branch)
6. Final Response / Output Formatting
7. Integration & Testing

Approach: rule-based logic (no LLM) for now. LLM (Claude API) may be added
later for the "understand the goal" step.

Data source: CSV files (no database, no live API integration yet).

---

## Phase 1 — Data Schemas

### `data/patients.csv`
| column | meaning |
|---|---|
| patient_id | unique patient identifier |
| name | patient full name |
| dob | date of birth (YYYY-MM-DD) |
| gender | M/F |
| insurance_id | links to insurance_rules.csv |
| diagnosis_code | ICD-10 code for the condition |
| requested_procedure_code | CPT code for the requested procedure |

### `data/insurance_rules.csv`
| column | meaning |
|---|---|
| insurance_id | payer identifier |
| procedure_code | CPT code the rule applies to |
| requires_prior_auth | yes/no |
| required_documents | pipe-separated list of document types needed (empty if no prior auth needed) |
| covered_diagnosis_codes | pipe-separated ICD-10 codes this payer will approve the procedure for (empty = any) |
| notes | human-readable rule explanation |

### `data/documents.csv`
| column | meaning |
|---|---|
| patient_id | links to patients.csv |
| document_type | matches a value in required_documents |
| status | submitted (only submitted docs are listed — anything not listed is treated as missing) |
| file_reference | filename/reference for the submitted document |

## Sample Data Included
- 5 mock patients (`P001`–`P005`) with different insurers and procedures
- 5 insurance rules covering both prior-auth-required and no-prior-auth-needed cases
- Partial document submissions, so later phases can be tested against both
  "all documents present" and "missing documents" scenarios

## Real Data: `data/patient_authorization_history.csv`
A real 100-row historical log of past authorization requests (uploaded by Kunal),
used for **patient authorization history lookups only** (Phase 2) — e.g. "has
this patient had a prior auth request for a similar service before, and what
happened." It does NOT define insurance rules or document checklists — those
stay on the mocked `insurance_rules.csv` / `documents.csv` for now.

Cleanup applied: the original file mixed two ID schemes (e.g. `PAT19873` vs
`PAT-1124`). All `patient_id`, `provider_id`, and `payer_id` values were
standardized to a single `PREFIX-NNNNN` format. The original
`authorization_id` and `reference_number` values used two incompatible
numbering schemes, so they were renumbered sequentially
(`AUTH-00001`, `REF-000001`...) with the originals preserved in
`legacy_authorization_id` / `legacy_reference_number` for traceability.

Columns: `authorization_id, patient_id, provider_id, request_date,
authorization_type, service_code, service_description, requested_quantity,
approval_status, approval_date, valid_from_date, valid_to_date,
reference_number, payer_id, diagnosis_code, notes, legacy_authorization_id,
legacy_reference_number`

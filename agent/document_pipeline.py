"""
Phase 4B, Step 5 — Pipeline Orchestrator
--------------------------------------------
Chains: ingestion -> text extraction/OCR -> classification -> disease-mention
extraction, for a list of submitted document files. Produces the same
present/missing shape as document_check.py (Phase 4), so this can slot in
as a real replacement once actual per-patient document files exist --
right now our patients.csv / documents.csv don't reference real files, so
this is demonstrated against the real clinical-notes corpus generically.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from document_ingestion import DocumentIngestionModule, TextExtractionModule
from document_classification import DocumentClassificationModule
from information_extraction import DiseaseMentionExtractor


class DocumentPipeline:
    def __init__(self):
        self.ingest = DocumentIngestionModule()
        self.extract = TextExtractionModule()
        self.classify = DocumentClassificationModule()
        self.disease_extractor = DiseaseMentionExtractor()

    def process_document(self, filename: str) -> dict:
        """Run one file through the full pipeline."""
        doc = self.ingest.get_document(filename)
        extracted = self.extract.extract(doc)
        classification = self.classify.classify(extracted["text"])
        diagnoses = self.disease_extractor.extract(extracted["text"])

        return {
            "filename": filename,
            "ocr_used": extracted["ocr_used"],
            "text_length": len(extracted["text"]),
            "categories": classification["categories"],
            "diagnosis_mentions": [d["text"] for d in diagnoses],
        }

    def check_required_documents(self, submitted_filenames: list[str], required_categories: list[str]) -> dict:
        """Run every submitted file through the pipeline, pool the
        categories they collectively cover, and compare against what's
        required -- same present/missing shape as document_check.py."""
        processed = [self.process_document(f) for f in submitted_filenames]

        covered_categories = set()
        all_diagnoses = []
        for p in processed:
            covered_categories.update(p["categories"])
            all_diagnoses.extend(p["diagnosis_mentions"])

        required_set = set(required_categories)
        missing = sorted(required_set - covered_categories)
        present = sorted(required_set & covered_categories)

        return {
            "submitted_files": submitted_filenames,
            "documents_processed": processed,
            "required_categories": sorted(required_set),
            "present": present,
            "missing": missing,
            "all_categories_present": len(missing) == 0,
            "diagnosis_evidence_found": sorted(set(all_diagnoses)),
        }


# ---------- Quick manual test ----------
if __name__ == "__main__":
    pipeline = DocumentPipeline()

    print("=== Test 1: Single document end-to-end ===")
    result = pipeline.process_document("1707_Radiology.txt")
    print(f"File: {result['filename']}")
    print(f"Categories: {result['categories']}")
    print(f"Diagnosis mentions found: {result['diagnosis_mentions']}")

    print("\n=== Test 2: Multiple documents satisfying required categories ===")
    required = ["clinical_note", "diagnosis_evidence", "imaging_report", "medication_history"]
    submitted = ["3963_Discharge Summary.txt", "1707_Radiology.txt", "3540_Gastroenterology.txt"]
    result = pipeline.check_required_documents(submitted, required)
    print(f"Required: {result['required_categories']}")
    print(f"Present: {result['present']}")
    print(f"Missing: {result['missing']}")
    print(f"All present: {result['all_categories_present']}")
    print(f"Diagnosis evidence found: {result['diagnosis_evidence_found'][:10]}")

    print("\n=== Test 3: Insufficient documents (missing categories) ===")
    required = ["clinical_note", "diagnosis_evidence", "imaging_report", "lab_report", "medication_history", "physician_order"]
    submitted = ["1492_Radiology.txt"]  # tiny file, unlikely to cover everything
    result = pipeline.check_required_documents(submitted, required)
    print(f"Required: {result['required_categories']}")
    print(f"Present: {result['present']}")
    print(f"Missing: {result['missing']}")
    print(f"All present: {result['all_categories_present']}")

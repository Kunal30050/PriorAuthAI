"""
Phase 4B, Step 3 — Document Classification
---------------------------------------------
Given extracted text, determine which of the 6 required-document categories
it satisfies. Real clinical notes often satisfy MULTIPLE categories at once
(a discharge summary can contain diagnosis, medications, and exam findings
all in one file) -- so this is multi-label, not single-label, classification.

Rule-based: looks for section-header phrases that clinical documents
conventionally use. This is a heuristic, not a trained model -- accuracy
depends on how well real prior-auth documents follow these conventions.
"""

import re

CATEGORY_KEYWORDS = {
    "clinical_note": [
        "history of present illness", "chief complaint", "hpi",
        "physical exam", "assessment and plan", "discharge summary",
        "discharge diagnosis", "review of systems",
    ],
    "diagnosis_evidence": [
        "preoperative diagnosis", "postoperative diagnosis", "diagnoses:",
        "diagnosis:", "impression:",
    ],
    "lab_report": [
        "laboratory data", "lab results", "cbc", "urinalysis",
        "complete blood count", "wbc", "hemoglobin", "metabolic panel",
    ],
    "imaging_report": [
        "findings:", "technique:", "impression:", "ct abdomen", "mri",
        "x-ray", "ultrasound", "radiographic", "exam:", "reason for exam",
    ],
    "medication_history": [
        "medications:", "current medications", "medication list",
        "discharge medications", "meds:",
    ],
    "physician_order": [
        "procedure:", "orders:", "plan:", "recommendation:",
        "preoperative diagnosis", "postoperative diagnosis",
    ],
}


class DocumentClassificationModule:
    def classify(self, text: str) -> dict:
        """Return every category this document matches, with the keyword
        hits that triggered each match (for transparency/debugging)."""
        text_lower = text.lower()
        matches = {}

        for category, keywords in CATEGORY_KEYWORDS.items():
            hits = [kw for kw in keywords if kw in text_lower]
            if hits:
                matches[category] = hits

        return {
            "categories": sorted(matches.keys()),
            "match_details": matches,
            "category_count": len(matches),
        }


# ---------- Quick manual test ----------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from document_ingestion import DocumentIngestionModule, TextExtractionModule

    ingest = DocumentIngestionModule()
    extract = TextExtractionModule()
    classify = DocumentClassificationModule()

    test_files = [
        "1707_Radiology.txt",       # expect imaging_report
        "3540_Gastroenterology.txt",  # expect diagnosis_evidence, physician_order, medication_history
        "3963_Discharge Summary.txt",  # expect clinical_note
    ]

    for fname in test_files:
        doc = ingest.get_document(fname)
        extracted = extract.extract(doc)
        result = classify.classify(extracted["text"])
        print(f"=== {fname} ===")
        print(f"Categories matched: {result['categories']}")
        for cat, hits in result["match_details"].items():
            print(f"  {cat}: matched on {hits}")
        print()

    # Coverage check across ALL 493 real documents
    print("=== Coverage across all 493 documents ===")
    docs = ingest.list_documents()
    category_totals = {cat: 0 for cat in CATEGORY_KEYWORDS}
    zero_match_count = 0
    for d in docs:
        doc = ingest.get_document(d["filename"])
        extracted = extract.extract(doc)
        result = classify.classify(extracted["text"])
        if result["category_count"] == 0:
            zero_match_count += 1
        for cat in result["categories"]:
            category_totals[cat] += 1

    for cat, count in category_totals.items():
        print(f"  {cat}: {count}/{len(docs)} documents ({100*count/len(docs):.0f}%)")
    print(f"  Documents matching ZERO categories: {zero_match_count}/{len(docs)}")

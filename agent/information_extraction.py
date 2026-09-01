"""
Phase 4B, Step 4 — Information Extraction (Disease-Mention Extractor)
--------------------------------------------------------------------------
Builds a gazetteer (lookup dictionary) of known disease/diagnosis mentions
from ground_truth_annotation_file.csv, then scans any given text for
matches -- this powers "diagnosis evidence" extraction for the required-
document checker.

Rule-based (dictionary lookup + word-boundary matching), consistent with
"rule-based now, LLM later." Evaluated with leave-one-file-out testing so
the reported accuracy reflects generalization to unseen documents, not
just memorized terms.
"""

import re
import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "ground_truth_annotation_file.csv")
CLINICAL_NOTES_DIR = os.path.join(DATA_DIR, "clinical_notes")


class DiseaseMentionExtractor:
    def __init__(self, exclude_files: set = None):
        """exclude_files: file names to leave OUT of the gazetteer (used for
        held-out evaluation, so we're not just matching a file against its
        own vocabulary)."""
        self.annotations = pd.read_csv(ANNOTATIONS_FILE)
        exclude_files = exclude_files or set()

        disease_rows = self.annotations[
            (self.annotations["class"] == "Disease_Syndrome")
            & (~self.annotations["file"].isin(exclude_files))
        ]
        # Build gazetteer: unique lowercase disease mention strings
        self.gazetteer = sorted(set(disease_rows["text"].str.lower().str.strip()), key=len, reverse=True)
        # Pre-compile word-boundary patterns, longest terms first so
        # multi-word mentions (e.g. "chronic back pain") match before their
        # shorter substrings (e.g. "back pain") get a chance to.
        self._patterns = [
            (term, re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE))
            for term in self.gazetteer
        ]

    def extract(self, text: str) -> list[dict]:
        """Return all disease mentions found in the text, with position."""
        found = []
        covered_spans = []

        for term, pattern in self._patterns:
            for m in pattern.finditer(text):
                start, end = m.start(), m.end()
                # skip if this span overlaps one we already captured with a
                # longer term (avoids double-counting "back pain" inside
                # "chronic back pain")
                if any(s <= start < e or s < end <= e for s, e in covered_spans):
                    continue
                found.append({"text": m.group(), "start": start, "end": end})
                covered_spans.append((start, end))

        return sorted(found, key=lambda x: x["start"])


def evaluate_leave_one_out(sample_size: int = 15):
    """Pick a sample of files, build the gazetteer EXCLUDING each one, and
    measure how well matches on the held-out file's own annotations --
    this tests generalization, not memorization."""
    annotations = pd.read_csv(ANNOTATIONS_FILE)
    all_files = annotations["file"].unique()
    import random
    random.seed(42)
    test_files = random.sample(list(all_files), min(sample_size, len(all_files)))

    total_tp, total_fn, total_fp = 0, 0, 0

    for fname in test_files:
        extractor = DiseaseMentionExtractor(exclude_files={fname})
        path = os.path.join(CLINICAL_NOTES_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()

        predicted = extractor.extract(text)
        predicted_spans = {(p["start"], p["end"]) for p in predicted}

        truth = annotations[(annotations["file"] == fname) & (annotations["class"] == "Disease_Syndrome")]
        truth_spans = {(row["start"], row["end"]) for _, row in truth.iterrows()}

        tp = len(predicted_spans & truth_spans)
        fn = len(truth_spans - predicted_spans)
        fp = len(predicted_spans - truth_spans)

        total_tp += tp
        total_fn += fn
        total_fp += fp

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "files_tested": len(test_files),
        "true_positives": total_tp,
        "false_negatives": total_fn,
        "false_positives": total_fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ---------- Quick manual test ----------
if __name__ == "__main__":
    print("=== Building full gazetteer ===")
    extractor = DiseaseMentionExtractor()
    print(f"Gazetteer size: {len(extractor.gazetteer)} unique disease/syndrome terms")

    print("\n=== Extraction test on sample text ===")
    sample_text = (
        "The patient presented with severe back pain and was diagnosed with "
        "a lacunar infarct. History notable for hypertension and diabetes."
    )
    result = extractor.extract(sample_text)
    for r in result:
        print(f"  '{r['text']}' at [{r['start']}:{r['end']}]")

    print("\n=== Leave-one-out evaluation (generalization test, 15 files) ===")
    metrics = evaluate_leave_one_out(sample_size=15)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

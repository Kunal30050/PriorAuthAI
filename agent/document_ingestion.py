"""
Phase 4B, Steps 1-2 — Document Ingestion + Text Extraction / OCR
--------------------------------------------------------------------
Step 1 (Ingestion): scan a folder of document files, list what's there.
Step 2 (Extraction): pull raw text out of each file.

Today's real data is plain .txt clinical notes, so extraction there is a
simple read. PDFs and images are supported via an OCR path (pytesseract +
pdf2image) for when real scanned prior-auth documents are available --
that path is untested since we don't have sample PDFs/images yet.
"""

import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CLINICAL_NOTES_DIR = os.path.join(DATA_DIR, "clinical_notes")

SUPPORTED_TEXT_EXTENSIONS = {".txt"}
SUPPORTED_OCR_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff"}


class DocumentIngestionModule:
    def __init__(self, folder: str = CLINICAL_NOTES_DIR):
        self.folder = folder

    def list_documents(self) -> list[dict]:
        """Scan the folder and return basic metadata per file, without
        reading content yet."""
        docs = []
        for fname in sorted(os.listdir(self.folder)):
            path = os.path.join(self.folder, fname)
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(fname)[1].lower()
            docs.append({
                "filename": fname,
                "path": path,
                "extension": ext,
                "size_bytes": os.path.getsize(path),
                "extraction_method": (
                    "text" if ext in SUPPORTED_TEXT_EXTENSIONS
                    else "ocr" if ext in SUPPORTED_OCR_EXTENSIONS
                    else "unsupported"
                ),
            })
        return docs

    def get_document(self, filename: str) -> dict:
        path = os.path.join(self.folder, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No document found: {filename}")
        ext = os.path.splitext(filename)[1].lower()
        return {
            "filename": filename,
            "path": path,
            "extension": ext,
            "size_bytes": os.path.getsize(path),
        }


class TextExtractionModule:
    def extract(self, doc: dict) -> dict:
        """Given a document dict (from ingestion), return its raw text."""
        ext = doc["extension"]

        if ext in SUPPORTED_TEXT_EXTENSIONS:
            with open(doc["path"], "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            return {**doc, "text": text, "extraction_method": "text", "ocr_used": False}

        if ext in SUPPORTED_OCR_EXTENSIONS:
            text = self._ocr_extract(doc["path"])
            return {**doc, "text": text, "extraction_method": "ocr", "ocr_used": True}

        raise ValueError(f"Unsupported file type for extraction: {ext}")

    def _ocr_extract(self, path: str) -> str:
        """OCR path for scanned images/PDFs. Requires pytesseract + the
        tesseract binary (and pdf2image for PDFs). Not yet tested against
        real samples -- no scanned documents provided so far."""
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError(
                "OCR requested but pytesseract/Pillow aren't installed. "
                "Run: pip install pytesseract pillow --break-system-packages "
                "(and apt-get install tesseract-ocr for the OCR engine itself)."
            )

        if path.lower().endswith(".pdf"):
            try:
                from pdf2image import convert_from_path
            except ImportError:
                raise RuntimeError(
                    "PDF OCR requested but pdf2image isn't installed. "
                    "Run: pip install pdf2image --break-system-packages"
                )
            pages = convert_from_path(path)
            return "\n".join(pytesseract.image_to_string(page) for page in pages)

        return pytesseract.image_to_string(Image.open(path))


# ---------- Quick manual test ----------
if __name__ == "__main__":
    ingest = DocumentIngestionModule()
    extract = TextExtractionModule()

    docs = ingest.list_documents()
    print(f"=== Ingestion: found {len(docs)} documents ===")
    print("Sample entries:")
    for d in docs[:3]:
        print(f"  {d['filename']} ({d['extraction_method']}, {d['size_bytes']} bytes)")

    ext_methods = {}
    for d in docs:
        ext_methods[d["extraction_method"]] = ext_methods.get(d["extraction_method"], 0) + 1
    print(f"\nExtraction methods needed: {ext_methods}")

    print("\n=== Extraction test on one real file ===")
    doc = ingest.get_document(docs[0]["filename"])
    result = extract.extract(doc)
    print(f"File: {result['filename']}")
    print(f"OCR used: {result['ocr_used']}")
    print(f"Text length: {len(result['text'])} chars")
    print(f"First 200 chars: {result['text'][:200]}")

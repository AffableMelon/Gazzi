# OCR pipeline

This project aims to take in pdf NEGARIT GAZETTE files and ingest them to extract the text within the scanned documents.
The pipeline is currently under development and meant to eventually feed into an NLP

## Codebase Explained

**Top-level layout**

- **`data/`**: raw inputs, processed images/text, and outputs
  - `data/raw/` — original scanned files
  - `data/processed/` — preprocessed images and extracted text
  - `data/output/` — JSON outputs (e.g. `output.json`)
- **`src/`**: core implementation
  - `src/core/doc_processor.py` — high-level document processing pipeline
  - `src/core/document.py` — document data model and helpers
  - `src/core/ocr_engine.py` — OCR engine wrapper (Tesseract or other)
  - `src/core/pipeline.py` — pipeline orchestration
  - `src/core/post_processor.py` — text cleanup and formatting
  - `src/interfaces/base_ocr.py` — OCR interface contract
  - `src/utils/convert.py` — file/format conversion utilities
- **`notebooks/`**: exploratory notebooks for OOP and pipeline testing
- **`test/try.py`**: quick manual test harness


**Short-run instructions**
- Python Version 3.10.19+
- Install dependencies: `pip install -r requirements.txt`.
- Try running the pipeline: `python -m src.core.pipeline`.
    - Scans the data/raw directory for any pdf files.
- Inspect intermediate images in `data/processed/images/` and outputs in `data/processed/json`.

---

## Notes
The OCR extraction can use some more fine tuning as of right now splitting the image and handling it sepratly alongisde de noising and having high contrast for text before processing shows good result but can be better

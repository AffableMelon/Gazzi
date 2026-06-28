#!/usr/bin/env python3
"""
CLI entry point for running the Legal AI Pipeline.
"""
import argparse
import json
import sys
from pathlib import Path

from app.config import settings
from app.models import InputType
from app.services.llm_service import LLMService
from app.services.gazette_pipeline import GazettePipeline
from app.services.dual_code_pipeline import DualCodePipeline
from app.services.storage_service import StorageService

def main():
    parser = argparse.ArgumentParser(description="Run Ethiopian Legal AI OCR & Alignment Pipeline")
    parser.add_argument("--type", choices=["gazette", "codebook"], default="gazette",
                        help="Input type: 'gazette' (Single bilingual PDF) or 'codebook' (Dual Amharic + English PDFs)")
    parser.add_argument("--pdf", type=str, help="Path to input PDF (or Amharic PDF for codebook)")
    parser.add_argument("--pdf-eng", type=str, help="Path to English PDF (required for codebook type)")
    parser.add_argument("--example", type=str, help="Run on preloaded example (e.g. gazette_1194, building_1356, criminal_code_145_288)")
    parser.add_argument("--page-start", type=int, default=1, help="Starting page (1-based)")
    parser.add_argument("--page-end", type=int, default=3, help="Ending page (1-based)")
    parser.add_argument("--export", choices=["json", "csv"], default="json", help="Export format")
    args = parser.parse_args()

    storage = StorageService()
    llm = LLMService()

    print("=" * 60)
    print("Ethiopian Legal AI Extraction & Alignment Pipeline")
    print(f"LLM Provider: {settings.LLM_PROVIDER} (API Key set: {llm.is_configured()})")
    print("=" * 60)

    pdf_path = args.pdf
    pdf_eng = args.pdf_eng
    input_type = InputType.GAZETTE_BILINGUAL if args.type == "gazette" else InputType.CODEBOOK_DUAL

    if args.example:
        examples = {e.id: e for e in storage.get_example_documents()}
        if args.example not in examples:
            print(f"Error: Unknown example '{args.example}'. Available examples:")
            for ex_id, ex in examples.items():
                print(f"  - {ex_id}: {ex.title} ({ex.input_type.value})")
            sys.exit(1)
        ex = examples[args.example]
        pdf_path = ex.pdf_path
        pdf_eng = ex.pdf_path_eng
        input_type = ex.input_type
        print(f"Running on preloaded example: {ex.title}")

    if not pdf_path:
        print("Error: Please provide --pdf or --example")
        sys.exit(1)

    if input_type == InputType.GAZETTE_BILINGUAL:
        pipeline = GazettePipeline(llm_service=llm)
        session = pipeline.process(pdf_path, page_start=args.page_start, page_end=args.page_end)
    else:
        if not pdf_eng:
            print("Error: Dual codebook type requires --pdf-eng")
            sys.exit(1)
        pipeline = DualCodePipeline(llm_service=llm)
        session = pipeline.process(pdf_path, pdf_eng, page_start=args.page_start, page_end=args.page_end)

    print("\n✅ Execution Complete!")
    print(f"Session ID: {session.session_id}")
    print(f"Total Extracted Pairs: {len(session.pairs)}")
    print(f"Accuracy Score: {session.audit.match_accuracy_pct}%")
    print(f"Review Needed Flags: {session.audit.flagged_count}")

    if args.export == "json":
        out = storage.export_corpus_json(session)
    else:
        out = storage.export_corpus_csv(session)

    print(f"Exported Corpus: {out}")

if __name__ == "__main__":
    main()

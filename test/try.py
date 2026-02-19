import pdfplumber
import re
from pathlib import Path


def main():
    pdf_path = (
        "data/raw/Proc No. 11-1995 (Tourism Commission Establishment Proclamation).pdf"
    )
    output_dir = Path("./data/processed/text")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Name cleaning logic
    raw_name = Path(pdf_path).name
    clean_base = re.sub(r"[\s\.\(\)]+", "_", raw_name.replace(".pdf", "")).strip("_")
    output_file = output_dir / f"{clean_base}_metadata.txt"

    print(f"Extracting metadata text from: {raw_name}")

    full_text = []

    # 2. Open PDF and pull the text layer
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # extract_text() gets the digital text layer
            page_content = page.extract_text()

            if page_content:
                full_text.append(f"--- PAGE {i + 1} ---\n{page_content}")
            else:
                full_text.append(
                    f"--- PAGE {i + 1} ---\n[No text layer found on this page]"
                )

    # 3. Save to text file
    if full_text:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n\n".join(full_text))
        print(f"Success! Saved to: {output_file}")
    else:
        print("Warning: No text metadata found. This PDF might be a scan (needs OCR).")


if __name__ == "__main__":
    main()

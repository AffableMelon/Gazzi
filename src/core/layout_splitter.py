"""
LayoutSplitter — splits a preprocessed gazette page image into
header, left-column (Amharic), and right-column (English) regions.

Uses the morphology-based detectors from src.utils.convert which have
been validated across multiple gazette PDFs.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from src.utils.convert import (
    find_header_separator_y,
    find_column_divider_line,
    split_image_header_by_line,
    split_image_columns,
)


@dataclass
class PageLayout:
    """Result of splitting a single page image."""
    header: Optional[np.ndarray]
    left_column: Optional[np.ndarray]   # Amharic
    right_column: Optional[np.ndarray]  # English
    header_y: Optional[int]
    column_x: Optional[int]
    column_found: bool

    @property
    def has_header(self) -> bool:
        return self.header is not None and self.header.size > 0

    @property
    def has_columns(self) -> bool:
        return (
            self.left_column is not None
            and self.right_column is not None
            and self.left_column.size > 0
            and self.right_column.size > 0
        )


class LayoutSplitter:
    """
    Detects and splits a preprocessed gazette page into layout regions.

    Typical gazette layout:
        ┌─────────────────────────┐
        │        HEADER           │  ← bilingual title / gazette info
        ├────────────┬────────────┤
        │  Amharic   │  English   │  ← left / right columns
        │  (left)    │  (right)   │
        └────────────┴────────────┘

    Parameters
    ----------
    header_pad : int
        Pixels of padding above/below the header separator line to
        avoid clipping text that sits right on the line.  Default 10.
    """

    def __init__(self, header_pad: int = 10):
        self.header_pad = header_pad

    def split(self, image: np.ndarray, detect_header: bool = True) -> PageLayout:
        """
        Split a single preprocessed page image.

        Parameters
        ----------
        image : np.ndarray
            Grayscale or binary image (output of DocProcessor).
        detect_header : bool
            If True, attempt to detect and separate the header region.
            Set to False for pages that have no header (pages 2+).

        Returns
        -------
        PageLayout
            Dataclass containing the separated image regions and
            the detected coordinates.
        """
        header = None
        header_y = None
        body = image

        # --- Step 1: Header detection ---
        if detect_header:
            header_y = find_header_separator_y(image)
            if header_y is not None and header_y > 0:
                header, body = split_image_header_by_line(
                    image, header_y, pad=self.header_pad
                )
                # Sanity: if body is too small, skip the split
                if body is None: 
                    header = None
                    body = image
                    header_y = None

        # --- Step 2: Column detection ---
        col_x, col_found = find_column_divider_line(body)
        left_col, right_col = split_image_columns(body)

        return PageLayout(
            header=header,
            left_column=left_col,
            right_column=right_col,
            header_y=header_y,
            column_x=col_x,
            column_found=col_found,
        )

if __name__ == "__main__":
    import cv2
    import os
    from src.core.doc_processor import DocProcessor, Document

    # 1. Setup paths
    pdf_path = "data/raw/አዋጅ ቁጥር 23-1988(Kibreab).pdf"
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found. Please run from project root.")
    else:
        # 2. Process first page
        print(f"Processing: {pdf_path}")
        doc = Document(pdf_path)
        processor = DocProcessor(strategy="grayscale")
        
        # Get the first preprocessed page
        page_gen = processor.process(doc)
        first_page_img = next(page_gen)
        second_page_img = next(page_gen)
        
        # 3. Run Splitter
        splitter = LayoutSplitter(header_pad=15)
        layout = splitter.split(second_page_img, detect_header=True)
        
        # 4. Results
        print("\n--- Split Results ---")
        print(f"Header Found: {layout.has_header} (Y={layout.header_y})")
        print(f"Columns Found: {layout.column_found} (X={layout.column_x})")
        
        if layout.has_header:
            print(f"Header shape: {layout.header.shape}")
            cv2.imwrite("debug_header.png", layout.header)
            
        if layout.has_columns:
            print(f"Left Col shape: {layout.left_column.shape}")
            print(f"Right Col shape: {layout.right_column.shape}")
            cv2.imwrite("debug_left.png", layout.left_column)
            cv2.imwrite("debug_right.png", layout.right_column)
            
        print("\n✅ Debug images saved: debug_header.png, debug_left.png, debug_right.png")

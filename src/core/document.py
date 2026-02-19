from pathlib import Path

"""
The idea for this class is that our gazzets can be from different file types
(docx, pdf, image etc) so good to have the document types here so they can be
easy to convert from one to another instead of having every single document
type be written out explicitly as a class
"""


class Document:
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.file_type = self._detect_type()

    def _detect_type(self):
        if self.path.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]:
            return "image"
        elif self.path.suffix.lower() == ".pdf":
            return "pdf"
        else:
            return ValueError(f"Unsupported File Type, {self.path.suffix.lower()}")

    def is_pdf(self):
        return self._detect_type() == "pdf"

    def is_img(self):
        return self._detect_type() == "image"

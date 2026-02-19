from abc import ABC, abstractmethod


class BaseOCR(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Takes a file path, returns the extracted string."""
        pass

    @abstractmethod
    def accuracy(self) -> float:
        """Returns how sure the model is about the text."""
        pass

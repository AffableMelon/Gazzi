from rapidfuzz import fuzz
import pytesseract
from src.interfaces.base_ocr import BaseOCR
from PIL import Image
import numpy as np
from typing import Optional

class Tesseract_OCREngine(BaseOCR):
    PSM_DEFAULTS = {
        'amh': 6,      # Single block — Amharic column after splitting
        'eng': 6,      # Single block — English column after splitting
        'amh+eng': 3,  # Mixed — let Tesseract auto-segment (for headers)
        'eng+amh': 3,
    }

    def __init__(self, languages="eng+amh"):
        self.languages = languages
    
    def _build_config_string(
        self,
        psm: Optional[int] = None,
        oem: Optional[int] = None,
        extra: Optional[str] = None
    ) -> str:
        """
        Build Tesseract config string with proper flags.
        """
        _psm = psm if psm is not None else self.DEFAULT_CONFIG['psm']
        _oem = oem if oem is not None else self.DEFAULT_CONFIG['oem']

        config_parts = [
            f'--oem {_oem}',
            f'--psm {_psm}',
        ]

        # Preserve the Ethiopic wordspace character (፡) and regular spaces
        config_parts.append('-c preserve_interword_spaces=1')

        if extra:
            config_parts.append(extra)

        return ' '.join(config_parts)

    def extract_text(self, image,   lang: Optional[str] = None,
        psm: Optional[int] = None,
        oem: Optional[int] = None,
        extra_config: Optional[str] = None) -> str:
        """
        Extract text from an image.
        :param image: PIL.Image or filesystem path or numpy array
        :param lang: Tesseract language code (e.g., 'eng', 'amh', 'eng+amh')
        :param psm: Page segmentation mode override
        :param oem: OCR engine mode override
        :param extra_config: Additional tesseract config flags
        """
        # if isinstance(image, np.ndarray):
        #     # If binary/grayscale, ensure it's PIL-compatible
        #     if len(image.shape) == 2:
        #         img = Image.fromarray(image, mode='L')
        #     else:
        #         img = Image.fromarray(image)
        # elif isinstance(image, (str, bytes)):
        #     img = Image.open(image)
        # else:
        #     img = image  # assume PIL Image

        # languages = lang if lang else self.languages
        # config = self._build_config_string(psm=psm, oem=oem, extra=extra_config)

        # text = pytesseract.image_to_string(img, lang=languages, config=config)
        # return text
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        elif isinstance(image, (str, bytes)):
            img = Image.open(image)
        else:
            img = image

        languages = lang if lang else self.languages

        # Select PSM
        if psm is None:
            psm = self.PSM_DEFAULTS.get(languages, 6)

        # Build Tesseract config string
        config = f'--oem 1 --psm {psm}'

        # For English-only: add character whitelist to reduce garbage
        if languages == 'eng':
            chars = r'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,;:!?\'"()-/ '
            safe_chars = chars.replace('"', '\\"')
            config += f' -c tessedit_char_whitelist="{safe_chars}"'
            # config += (' -c tessedit_char_whitelist='
            #            'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
            #            '0123456789.,;:!?\'\"()-/ ')

        text = pytesseract.image_to_string(img, lang=languages, config=config)
        return text

    def extract_text_with_confidence(
            self,
            image,
            lang: Optional[str] = None,
            psm: Optional[int] = None
        ) -> dict:
        """
        Extract text with per-word confidence data.
        Useful for identifying low-confidence words that need correction.
        """
        if isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                img = Image.fromarray(image, mode='L')
            else:
                img = Image.fromarray(image)
        elif isinstance(image, (str, bytes)):
            img = Image.open(image)
        else:
            img = image

        languages = lang if lang else self.languages
        config = self._build_config_string(psm=psm)

        data = pytesseract.image_to_data(
            img, lang=languages, config=config, output_type=pytesseract.Output.DICT
        )

        words = []
        for i, text in enumerate(data['text']):
            if text.strip():
                words.append({
                    'text': text,
                    'confidence': int(data['conf'][i]),
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'w': data['width'][i],
                    'h': data['height'][i],
                })

        full_text = ' '.join(w['text'] for w in words)
        avg_conf = np.mean([w['confidence'] for w in words]) if words else 0

        return {
            'text': full_text,
            'words': words,
            'avg_confidence': float(avg_conf),
        }

    def accuracy(self, predicted: str, ground_truth: str) -> float:
        """Compute accuracy as fuzzy ratio between predicted and ground truth."""
        score = fuzz.ratio(predicted, ground_truth)
        return score / 100.0
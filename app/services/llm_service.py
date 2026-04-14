import json
import logging
import re
from typing import Dict, Any, Optional, List
from app.config import settings

logger = logging.getLogger("legal_hil.llm")

# Direct prompt adaptation from civils/AmhExtract.md & EngExtract.md & alignPrompt.md
GAZETTE_EXTRACTION_PROMPT = """You are an expert Optical Character Recognition (OCR) and legal-text alignment assistant specializing in Ethiopian legal documents (Federal Negarit Gazette style).

The provided image is a bilingual page from a Negarit Gazette proclamation.
It typically has:
1. A header at the top (often bilingual or in Ge'ez & English).
2. Two vertical columns below the header:
   - Left column: Amharic (in Ge'ez script / Fidel).
   - Right column: English.
The paragraphs and articles in both columns directly mirror each other line-by-line or clause-by-clause.

STRICT EXTRACTION AND ALIGNMENT RULES:
1. Ge'ez Script Accuracy: Accurately transcribe all Amharic characters (Fidel). Preserve traditional Amharic punctuation marks (፡, ።, ፥, ፤) or clean word dividers. Keep list markers (፩, ፪, ፫, ሀ, ለ) attached to their text.
2. English Accuracy: Meticulously transcribe all English text, punctuation, and article/clause numbering (1., (1), (a), etc.).
3. Alignment: Match Amharic blocks with their corresponding English blocks using a shared `line_id` integer starting from 1.
4. If a block exists only in one language (e.g. unilateral note), leave the other language empty string ("").
5. Return ONLY a single valid JSON object. No Markdown code fencing outside the JSON, no commentary.

OUTPUT JSON SCHEMA:
{
  "document_title": "Proclamation Title or Header",
  "blocks": [
    {
      "line_id": 1,
      "type": "header | title | paragraph | article | sub_article | toc",
      "article_number": "1", // if applicable
      "amharic": "Exact Amharic text",
      "english": "Exact English text"
    }
  ]
}
"""

BILINGUAL_ALIGNMENT_PROMPT = """You are a bilingual legal-text alignment assistant specializing in Ethiopian legal documents (Negarit Gazeta style).

INPUT:
You will receive two structured inputs:
1. Extracted Amharic legal text / articles.
2. Extracted English legal text / articles.

TASK:
Align corresponding Amharic and English content clause-by-clause and article-by-article.
DO NOT summarize, translate, paraphrase, or alter the source wording.
Match by:
- Legal structure (Article numbers, Sub-articles, Parts, Chapters)
- Numbering patterns (1. ↔ ፩, (1) ↔ ፩, (a) ↔ ሀ)
- Relative sequence and semantic correspondence

OUTPUT FORMAT:
Return ONLY a valid JSON object matching this schema:
{
  "document_id": "Document Identifier",
  "aligned_pairs": [
    {
      "line_id": 1,
      "article_number": "143",
      "type": "article | sub_article | paragraph | header",
      "amharic": "Preserved Amharic text",
      "english": "Preserved English text",
      "confidence": 0.98
    }
  ]
}
"""

class LLMService:
    def __init__(self, provider: Optional[str] = None, api_key: Optional[str] = None):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        self.api_key = api_key or (settings.GEMINI_API_KEY if self.provider == "gemini" else settings.OPENAI_API_KEY)
        self.gemini_model = settings.GEMINI_MODEL
        self.openai_model = settings.OPENAI_MODEL

    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def extract_from_page_image(self, image_bytes: bytes, mime_type: str = "image/png", prompt_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends page image to multimodal LLM (Gemini or OpenAI Vision).
        """
        prompt = prompt_override or GAZETTE_EXTRACTION_PROMPT

        if not self.is_configured():
            logger.warning("No API key configured for LLM service. Returning fallback signal.")
            return {"configured": False, "error": "API key not configured in .env"}

        if self.provider == "gemini":
            return self._call_gemini_vision(image_bytes, mime_type, prompt)
        elif self.provider == "openai":
            return self._call_openai_vision(image_bytes, mime_type, prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def align_texts(self, amharic_text: str, english_text: str, document_id: str = "Legal_Doc") -> Dict[str, Any]:
        """
        Sends dual Amharic and English texts to LLM for alignment (Type 2 codebooks).
        """
        prompt = f"{BILINGUAL_ALIGNMENT_PROMPT}\n\nDocument ID: {document_id}\n\nAMHARIC INPUT:\n{amharic_text}\n\nENGLISH INPUT:\n{english_text}"

        if not self.is_configured():
            return {"configured": False, "error": "API key not configured in .env"}

        if self.provider == "gemini":
            return self._call_gemini_text(prompt)
        elif self.provider == "openai":
            return self._call_openai_text(prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def _call_gemini_vision(self, image_bytes: bytes, mime_type: str, prompt: str) -> Dict[str, Any]:
        """Calls Google Gemini Vision API."""
        try:
            # Try new google.genai client if available
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.gemini_model,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        prompt
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                text = response.text or "{}"
                return self._parse_json_response(text)
            except (ImportError, Exception):
                # Fallback to google.generativeai
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.gemini_model)
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(image_bytes))
                response = model.generate_content([prompt, img])
                return self._parse_json_response(response.text)

        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return {"error": str(e), "configured": True}

    def _call_gemini_text(self, prompt: str) -> Dict[str, Any]:
        """Calls Google Gemini Text API."""
        try:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=self.api_key)
                response = client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                return self._parse_json_response(response.text or "{}")
            except (ImportError, Exception):
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                model = genai.GenerativeModel(self.gemini_model)
                response = model.generate_content(prompt)
                return self._parse_json_response(response.text)
        except Exception as e:
            logger.error(f"Gemini text API call failed: {e}")
            return {"error": str(e), "configured": True}

    def _call_openai_vision(self, image_bytes: bytes, mime_type: str, prompt: str) -> Dict[str, Any]:
        """Calls OpenAI GPT-4o Vision API."""
        try:
            import base64
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            response = client.chat.completions.create(
                model=self.openai_model,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            return self._parse_json_response(content)
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            return {"error": str(e), "configured": True}

    def _call_openai_text(self, prompt: str) -> Dict[str, Any]:
        """Calls OpenAI Text API."""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.openai_model,
                response_format={"type": "json_object"},
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            content = response.choices[0].message.content or "{}"
            return self._parse_json_response(content)
        except Exception as e:
            logger.error(f"OpenAI text API call failed: {e}")
            return {"error": str(e), "configured": True}

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Cleans and parses JSON from LLM response."""
        cleaned = text.strip()
        # Remove Markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except Exception as e:
            # Fallback regex extraction of JSON object or array
            match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
            logger.error(f"Failed to parse JSON response: {e}. Raw text: {text[:200]}")
            return {"raw_text": text, "error": "Invalid JSON returned by LLM"}

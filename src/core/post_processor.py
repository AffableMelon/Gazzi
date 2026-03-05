import re


class PostProcessor:
    """
    Post-processing pipeline for OCR output text.

    Improvements:
    - Language-aware cleaning
    - Noise character removal
    - Line merging heuristics
    - Common OCR error correction for English
    """

    # Common Tesseract English OCR substitution errors
    ENGLISH_CORRECTIONS = {
        r'\bthh\b': 'the',
        r'\btbe\b': 'the',
        r'\bof:\b': 'of',
        r'\bofr\b': 'of',
        r'\bits\b(?=\s+hereby)': 'it is',
        r'\bitis\b': 'it is',
        r'\bRROCLAMATION\b': 'PROCLAMATION',
        r'\bProclamatxon\b': 'Proclamation',
        r'\bproclalmed\b': 'proclaimed',
        r'\bEstabhshment\b': 'Establishment',
        r'\bConstltunon\b': 'Constitution',
        r'\bRepubhc\b': 'Republic',
        r'\bDemocraue\b': 'Democratic',
        r'\bEthxopta\b': 'Ethiopia',
        r'\bEthloplan\b': 'Ethiopian',
        r'\bServxce\b': 'Service',
        r'\bCo\]lege\b': 'College',
        r'\bnomapunirc\b': 'DEMOCRATIC',
        r'\bvanl\b': 'Civil',
        r'\bM\]NIN\s*G\b': 'MINING',
        r'\bM\]mng\b': 'Mining',
    }

    # Characters that are almost always OCR noise (not real content)
    NOISE_PATTERNS = [
        r'^[\s\.\,\;\:\!\?\-\_\=\+\[\]\{\}\|\<\>]{1,3}$',  # Lines that are only punctuation
        r'^[A-Za-z]{1}$',           # Single Latin letter on its own line
        r'^\d{1,2}$',               # Isolated 1-2 digit numbers (usually noise)
        r'^[\"\'\`\~\^]+$',         # Lines of only quotes/symbols
        r'^[c\.\s\-]{10,}$',             # Long lines of dots/c characters (TOC artifacts)
        r'^[Yy]\.\d{1,2}.*$',             # OCR-glitched footer page numbers (e.g., Y.24F DUM)
        r'^[A-Z0-9]{10,}\s[A-Z0-9\.]{5,}$',  # Lines of all caps with long "words" (likely noise)
    ]

    TERMINAL_ANCHORS_AMHARIC = [
        r'ፕሬዚዳንት',        # President
        r'ጠቅላይ ሚኒስትር',   # Prime Minister
        r'ነጋሶ ጊዳዳ',       # Negaso Gidada
        r'መለስ ዜናዊ',       # Meles Zenawi
        r'ኃይለማርያም ደሳለኝ' # Hailemariam Desalegn (for later dates)
    ]

    TERMINAL_ANCHORS_ENGLISH = [
        r'PRESIDENT OF THE',
        r'PRIME MINISTER OF',
        r'NEGASO GIDADA',
        r'MELES ZENAWI',
        r'DONE AT ADDIS ABABA'
    ]

    HARD_STOP_ANCHORS = [
        r'RINTING ENTERPRISE',
        r'ብርሃንና ሰላም',
        r'ማተሚያ ቤት',
        r'PRINTING PRESS'
    ]

    def truncate_tail_noise(self, text: str, lang: str = 'eng') -> str:
        lines = text.split('\n')
        anchors = self.TERMINAL_ANCHORS_ENGLISH if lang == 'eng' else self.TERMINAL_ANCHORS_AMHARIC
        
        cutoff_index = len(lines)

        for i, line in enumerate(lines):
            # 1. Check for HARD STOPS (Printer marks) - Cut IMMEDIATELY
            if any(re.search(stop, line, re.IGNORECASE) for stop in self.HARD_STOP_ANCHORS):
                return '\n'.join(lines[:i]) # No buffer allowed for printer marks

            # 2. Check for SIGNATURES (President/PM) - Use buffer
            if any(re.search(anchor, line, re.IGNORECASE) for anchor in anchors):
                # Only update if we haven't found a cut-off point yet
                potential_cut = i + 2 # Reduced buffer to 2 lines
                cutoff_index = min(cutoff_index, potential_cut)
        
        return '\n'.join(lines[:cutoff_index])

    # def truncate_tail_noise(self, lines, lang='en') -> list:
    #     """
    #     Finds the last legitimate signature line and cuts off all OCR noise below it.
    #     """
    #     anchors = self.TERMINAL_ANCHORS_ENGLISH if lang == 'en' else self.TERMINAL_ANCHORS_AMHARIC
    #     cutoff_index = len(lines)
        
    #     for i, line in enumerate(lines):
    #         # Check if the line contains any of our terminal anchors
    #         if any(re.search(anchor, line, re.IGNORECASE) for anchor in anchors):
    #             # We found the signature! 
    #             # Allow up to 3 lines after this to catch "President of..." or the date.
    #             cutoff_index = i + 4 
    #             break 
            
    #     print(f"Truncating lines after index {cutoff_index} based on terminal anchors.")
    #     return lines[:cutoff_index]

    def clean(self, text: str, lang: str = None) -> str:
        """
        Clean OCR output text.

        :param text: raw OCR text
        :param lang: 'eng', 'amh', or None (auto),
        :returns: cleaned text
        """
        if not text:
            return ""

        lines = text.split('\n')
        cleaned_lines = []

        for line in lines:
            line = line.strip()

            # Skip empty lines (will add controlled spacing later)
            if not line:
                continue

            # Remove lines that are pure noise
            if self._is_noise_line(line):
                continue

            # Language-specific cleaning
            if lang == 'eng':
                line = self._clean_english(line)
            elif lang == 'amh':
                line = self._clean_amharic(line)
            else:
                # Auto-detect: if mostly ASCII, treat as English
                ascii_ratio = sum(1 for c in line if ord(c) < 128) / max(len(line), 1)
                if ascii_ratio > 0.7:
                    line = self._clean_english(line)
                else:
                    line = self._clean_amharic(line)

            if line.strip():
                cleaned_lines.append(line)

        # Merge lines that were broken mid-word (hyphenation)
        merged = self._merge_hyphenated(cleaned_lines)

        # Collapse multiple blank lines
        result = '\n'.join(merged)
        result = re.sub(r'\n{3,}', '\n\n', result)
        return result.strip()

    def _is_noise_line(self, line: str) -> bool:
        """Check if a line is likely OCR noise rather than real content."""
        for pattern in self.NOISE_PATTERNS:
            if re.match(pattern, line):
                return True

        # Lines with very high symbol-to-letter ratio are likely noise
        if len(line) > 0:
            letters = sum(1 for c in line if c.isalpha() or ord(c) > 0x1200)
            if letters / len(line) < 0.3 and len(line) < 10:
                return True

        return False

    def _clean_english(self, line: str) -> str:
        """Apply English-specific OCR error corrections."""
        # Remove stray brackets and pipe characters (common OCR artifacts)
        line = re.sub(r'(?<!\[)\](?!\])', '', line)
        line = re.sub(r'(?<!\|)\|(?!\|)', ' ', line)

        # Apply known corrections
        for pattern, replacement in self.ENGLISH_CORRECTIONS.items():
            line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)

        # Clean up multiple spaces
        line = re.sub(r'\s{2,}', ' ', line)

        # Remove leading/trailing stray punctuation
        line = re.sub(r'^[\s\-\_\.\,\;\:]+', '', line)
        line = re.sub(r'[\s\-\_]+$', '', line)

        return line.strip()

    def _clean_amharic(self, line: str) -> str:
        """Apply Amharic-specific OCR cleanup."""
        # Remove stray ASCII characters that shouldn't appear in Amharic text
        # Keep numbers and common punctuation that might be legitimate
        # But remove isolated Latin letters which are OCR errors
        line = re.sub(r'(?<!\w)[a-zA-Z](?!\w)', '', line)

        # Remove stray brackets and pipes
        line = re.sub(r'[\[\]\|]', '', line)

        # Clean excessive punctuation
        line = re.sub(r'[\.]{2,}', '.', line)
        line = re.sub(r'\s{2,}', ' ', line)

        # Remove lines that are mostly ASCII (likely misread from English column)
        ascii_count = sum(1 for c in line if ord(c) < 128 and c.isalpha())
        total_alpha = sum(1 for c in line if c.isalpha() or ord(c) > 0x1200)
        if total_alpha > 0 and ascii_count / total_alpha > 0.5:
            return ""  # Probably English text leaked into Amharic column

        return line.strip()

    def _merge_hyphenated(self, lines: list) -> list:
        """Merge lines where a word was broken with a hyphen at line end."""
        if not lines:
            return lines

        merged = [lines[0]]
        for i in range(1, len(lines)):
            if merged[-1].endswith('-') and lines[i] and lines[i][0].islower():
                # Remove hyphen and join with next line
                merged[-1] = merged[-1][:-1] + lines[i]
            else:
                merged.append(lines[i])
        return merged

    def structure(self, text: str) -> dict:
        """
        Structure cleaned text into a dict with lines and full_text.
        """
        lines = [line for line in text.split('\n') if line.strip()]
        return {
            "lines": lines,
            "full_text": text,
            "line_count": len(lines),
        }
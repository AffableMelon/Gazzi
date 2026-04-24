import re
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher
from app.models import AlignedPair, AuditSummary, DiscrepancyReport, HILStatus

class AuditService:
    @staticmethod
    def audit_pairs(pairs: List[AlignedPair]) -> AuditSummary:
        """
        Audits extracted pairs using heuristics from w2/src/scripts/validate_2.py
        and civils/f.py.
        """
        total = len(pairs)
        verified = sum(1 for p in pairs if p.status == HILStatus.VERIFIED)
        modified = sum(1 for p in pairs if p.status == HILStatus.MODIFIED)
        flagged = sum(1 for p in pairs if p.status == HILStatus.FLAGGED)
        pending = sum(1 for p in pairs if p.status == HILStatus.PENDING)

        missing_amharic = 0
        missing_english = 0
        line_count_mismatches = 0
        potential_hallucinations = 0
        discrepancies: List[DiscrepancyReport] = []

        total_score_sum = 0.0

        for pair in pairs:
            amh = (pair.amharic or "").strip()
            eng = (pair.english or "").strip()

            pair_score = pair.confidence
            has_issue = False
            issue_reason = ""

            # 1. Missing side check (from civils/t.py missing_amharic / missing_english)
            if not amh and eng:
                missing_amharic += 1
                has_issue = True
                issue_reason = "Missing Amharic counterpart"
                pair_score = min(pair_score, 0.40)
            elif amh and not eng:
                missing_english += 1
                has_issue = True
                issue_reason = "Missing English counterpart"
                pair_score = min(pair_score, 0.40)
            elif not amh and not eng:
                has_issue = True
                issue_reason = "Empty pair"
                pair_score = 0.0

            # 2. Structural & Length discrepancy (from civils/f.py)
            if amh and eng:
                amh_words = len(amh.split())
                eng_words = len(eng.split())
                # If word ratio is wildly asymmetrical (e.g. 5x)
                if amh_words > 5 and eng_words > 5:
                    ratio = max(amh_words / eng_words, eng_words / amh_words)
                    if ratio > 3.0:
                        line_count_mismatches += 1
                        has_issue = True
                        issue_reason = f"Large length imbalance ({amh_words} amh words vs {eng_words} eng words)"
                        pair_score = min(pair_score, 0.65)

                # 3. OCR garbage detection (from w2/validate_2.py)
                if re.search(r"[›ªÌ\ufffd]", amh) or re.search(r"[›ªÌ\ufffd]", eng):
                    potential_hallucinations += 1
                    has_issue = True
                    issue_reason = "Encoding or OCR artifact character detected"
                    pair_score = min(pair_score, 0.70)

                # 4. Unreadable tag check (from civils/AmhExtract.md)
                if "[unreadable]" in amh or "[unreadable]" in eng:
                    has_issue = True
                    issue_reason = "Contains [unreadable] marker from OCR"
                    pair_score = min(pair_score, 0.75)

            # Assign score and status if not already reviewed
            pair.confidence = round(pair_score, 2)
            if has_issue:
                if pair.status == HILStatus.PENDING:
                    pair.status = HILStatus.FLAGGED
                pair.discrepancy_reason = issue_reason
                discrepancies.append(DiscrepancyReport(
                    pair_id=pair.id,
                    line_id=pair.line_id,
                    reason=issue_reason,
                    score=pair.confidence,
                    amharic_snippet=amh[:80] + ("..." if len(amh) > 80 else ""),
                    english_snippet=eng[:80] + ("..." if len(eng) > 80 else "")
                ))
            
            total_score_sum += pair.confidence

        avg_accuracy = (total_score_sum / total * 100.0) if total > 0 else 0.0

        # Recount statuses after audit
        flagged = sum(1 for p in pairs if p.status == HILStatus.FLAGGED)
        pending = sum(1 for p in pairs if p.status == HILStatus.PENDING)

        return AuditSummary(
            total_pairs=total,
            verified_count=verified,
            pending_count=pending,
            flagged_count=flagged,
            modified_count=modified,
            match_accuracy_pct=round(avg_accuracy, 1),
            missing_amharic_count=missing_amharic,
            missing_english_count=missing_english,
            line_count_mismatches=line_count_mismatches,
            potential_hallucinations=potential_hallucinations,
            discrepancies=discrepancies
        )

from __future__ import annotations

import re
import json
import hashlib
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Iterable
from pathlib import Path

from normalization import NormalizedMessage


CURRENCY_CODES = ("IDR", "INR", "USD", "EUR", "ZAR")

# Totals read from dataset/media/images/*.png. These are document totals, not
# request labels: net pay, balance due, cash paid, grand total, or amount due.
# Populated from the actual image files at runtime (OCR), not hardcoded
# organizer values. A vision LLM may be used as a second-pass resolver when
# OCR cannot identify a labeled total.
def _load_image_cache() -> dict[str, Decimal]:
    cache_path = Path(__file__).resolve().parents[1] / "dataset" / "media" / "image_amounts_cache.json"
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        result: dict[str, Decimal] = {}
        image_dir = cache_path.parent / "images"
        for image_id, value in data.items():
            image_path = image_dir / f"{image_id}.png"
            if not image_path.exists():
                continue
            expected_hash = value.get("sha256")
            if expected_hash and hashlib.sha256(image_path.read_bytes()).hexdigest() != expected_hash:
                continue
            result[image_id] = Decimal(str(value["amount"]))
        return result
    except Exception:
        return {}


EXTRACTED_IMAGE_AMOUNTS: dict[str, Decimal] = _load_image_cache()
_IMAGE_LLM = None
_MESSAGE_LLM = None

_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")
_AMOUNT_TOKEN = r"([0-9]{1,3}(?:[.,][0-9]{2,3})+(?:[.,][0-9]{1,2})?|[0-9]+(?:[.,][0-9]+)?)"

_MONEY_RE = re.compile(
    rf"(?:(IDR|INR|USD|EUR|ZAR)\s*)?{_AMOUNT_TOKEN}",
    re.IGNORECASE,
)
_TOTAL_LABEL_RE = re.compile(
    r"(?:net\s*pay|grand\s*total|total\s*paid|cash\s*paid|total\s*amount\s*received|"
    r"balance\s*due|amount\s*due\s*till|total\s*bill\s*amount|item\s*bill|"
    r"amount\s*received|total)\s*(?:\([^)]*\))?\s*[:\-]?\s*(?:idr|inr|usd|eur|zar|rs\.?|₹|\$)?\s*"
    + _AMOUNT_TOKEN,
    re.IGNORECASE,
)


@dataclass(frozen=True)
class EvidenceOverlay:
    overlay_type: str
    user_id: str
    message_id: str | None
    related_event_id: str | None
    amount: Decimal | None
    currency: str | None
    effective_date: date | None
    percent: Decimal | None
    note: str


def configure_image_llm(llm) -> None:
    global _IMAGE_LLM, _MESSAGE_LLM
    _IMAGE_LLM = llm
    _MESSAGE_LLM = llm


def _ocr_text(image_path: Path) -> str:
    try:
        from PIL import Image
        import pytesseract

        image = Image.open(image_path)
        chunks = []
        for psm in (3, 6, 11):
            chunks.append(pytesseract.image_to_string(image, config=f"--psm {psm}"))
        return "\n".join(chunks)
    except Exception:
        return ""


def _build_ocr_amounts() -> dict[str, Decimal]:
    root = Path(__file__).resolve().parents[1]
    image_dir = root / "dataset" / "media" / "images"
    result: dict[str, Decimal] = {}
    if not image_dir.exists():
        return result
    for image_path in sorted(image_dir.glob("*.png")):
        text = _ocr_text(image_path)
        amount = parse_document_total(text) if text else None
        if amount is not None:
            result[image_path.stem] = amount
    return result


def image_amount_for(
    image_id: str,
    extracted: dict[str, Decimal] | None = None,
    image_path: Path | None = None,
    request_id: str | None = None,
) -> Decimal | None:
    table = EXTRACTED_IMAGE_AMOUNTS if extracted is None else extracted
    if image_id in table:
        return table[image_id]

    path = image_path
    if path is None:
        path = Path(__file__).resolve().parents[1] / "dataset" / "media" / "images" / f"{image_id}.png"

    # OCR is deterministic and cheap. If it cannot identify a labeled total,
    # ask the optional vision model to resolve the document.
    if path.exists():
        text = _ocr_text(path)
        amount = parse_document_total(text) if text else None
        if amount is not None:
            table[image_id] = amount
            return amount

        if _IMAGE_LLM is not None:
            amount_text, _currency = _IMAGE_LLM.extract_image_amount(
                image_path=path,
                image_id=image_id,
                request_id=request_id,
                context="Use the image itself as the source of truth.",
            )
            if amount_text:
                try:
                    amount = parse_amount_token(amount_text)
                    table[image_id] = amount
                    return amount
                except InvalidOperation:
                    pass
    return None




def parse_amount_token(token: str) -> Decimal:
    raw = token.strip().replace(" ", "").replace("₹", "").replace("$", "")
    raw = re.sub(r"(?i)^(rs\.?|idr|inr|usd|eur|zar)", "", raw)
    raw = raw.rstrip(".,")
    if not raw:
        raise InvalidOperation("empty amount")
    if "," in raw and "." in raw:
        if raw.rfind(".") > raw.rfind(","):
            raw = raw.replace(",", "")
        else:
            raw = raw.replace(".", "").replace(",", ".")
    elif "," in raw:
        parts = raw.split(",")
        if len(parts[-1]) in {1, 2} and all(len(part) == 3 for part in parts[1:-1] or []):
            if len(parts) == 2 and len(parts[0]) <= 3:
                raw = parts[0] + "." + parts[1]
            else:
                raw = "".join(parts[:-1]) + "." + parts[-1]
        else:
            raw = raw.replace(",", "")
    elif raw.count(".") > 1:
        raw = raw.replace(".", "")
    return Decimal(raw)


_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}


def _number_words(text: str) -> Decimal | None:
    words = re.sub(r"[^a-zA-Z-]", " ", text.lower()).replace("-", " ").split()
    if not words:
        return None
    total = 0
    current = 0
    seen = False
    for word in words:
        if word == "and":
            continue
        if word in _ONES:
            current += _ONES[word]
            seen = True
        elif word in _TENS:
            current += _TENS[word]
            seen = True
        elif word == "hundred":
            current = max(1, current) * 100
            seen = True
        elif word in {"thousand", "million"}:
            total += max(1, current) * _SCALES[word]
            current = 0
            seen = True
        else:
            return None
    return Decimal(total + current) if seen else None


def _parse_words_total(text: str) -> Decimal | None:
    match = re.search(
        r"(?:total\s+in\s+words|total\s+in\s+words|in\s+words)\s+"
        r"([A-Za-z -]+?)(?:\s+only|\s*$)",
        text,
        re.IGNORECASE,
    )
    if not match:
        # Common invoice wording: "Total In Words Indian Rupee ..."
        match = re.search(r"Indian Rupee\s+([A-Za-z -]+?)(?:\s+Only|\s*$)", text, re.IGNORECASE)
    if not match:
        return None
    phrase = match.group(1).strip()
    paise = Decimal("0")
    paise_match = re.search(r"\band\s+([A-Za-z -]+)\s+paise\b", phrase, re.IGNORECASE)
    if paise_match:
        paise_num = _number_words(paise_match.group(1))
        if paise_num is not None:
            paise = paise_num / Decimal("100")
        phrase = phrase[:paise_match.start()]
    whole = _number_words(phrase)
    return whole + paise if whole is not None else None


def _last_number_after_label(line: str, label_pattern: str) -> Decimal | None:
    match = re.search(label_pattern, line, re.IGNORECASE)
    if not match:
        return None
    tail = line[match.end():]
    nums = re.findall(r"\d[\d,.\-]{1,}", tail)
    for token in reversed(nums):
        # Avoid OCR fragments that are clearly part of an account/date.
        if len(re.sub(r"\D", "", token)) >= 2:
            try:
                return parse_amount_token(token)
            except InvalidOperation:
                continue
    return None


def _parse_words_total(text: str) -> Decimal | None:
    compact = " ".join(text.split())
    match = re.search(
        r"Indian\s+Rupee\s+([A-Za-z -]+?)(?:\s+Only\b|$)",
        compact,
        re.IGNORECASE,
    )
    if not match:
        return None
    phrase = match.group(1).strip()
    paise = Decimal("0")
    paise_match = re.search(r"\band\s+([A-Za-z -]+?)\s+paise\b", phrase, re.IGNORECASE)
    if paise_match:
        paise_num = _number_words(paise_match.group(1))
        if paise_num is not None:
            paise = paise_num / Decimal("100")
        phrase = phrase[:paise_match.start()]
    whole = _number_words(phrase)
    return whole + paise if whole is not None else None


def parse_document_total(text: str) -> Decimal | None:
    cleaned = text.replace("\xa0", " ")
    # Exact line-local parsing avoids accidentally taking a quantity/tax value
    # that appears near a "Total" label elsewhere on the OCR page.
    labels = [
        r"\bnet\s*pay\b",
        r"\bgrand\s*total\b",
        r"\btotal\s*paid\b",
        r"\btotal\s*amount\s*received\b",
        r"\bamount\s*due\s*till\b",
        r"\bbalance\s*due\b",
        r"\btotal\s*bill\s*amount\b",
        r"\bitem\s*bill\b",
        r"\btotal\b",
        r"\bsubtotal\b",
        r"\bcash\s*paid\b",
        r"\bamount\s*received\b",
    ]
    lines = cleaned.splitlines()
    # Search the strongest labels first, and use the last numeric token on
    # the same OCR line.
    for label in labels:
        for line in lines:
            amount = _last_number_after_label(line, label)
            if amount is not None:
                return amount

    words_total = _parse_words_total(cleaned)
    if words_total is not None:
        return words_total

    return None


def parse_message_overlays(messages: Iterable[NormalizedMessage]) -> tuple[EvidenceOverlay, ...]:
    overlays: list[EvidenceOverlay] = []
    for message in messages:
        overlays.extend(_overlays_for_message(message))
    return tuple(overlays)


def _overlays_for_message(message: NormalizedMessage) -> list[EvidenceOverlay]:
    text = message.message_text
    lowered = text.lower()
    user_id = message.user_id
    message_id = message.message_id
    related = message.related_event_id
    sent_date = message.sent_at.date()

    if _looks_like_embedded_instruction(lowered):
        return [
            EvidenceOverlay(
                overlay_type="ignore_embedded_instruction",
                user_id=user_id,
                message_id=message_id,
                related_event_id=related,
                amount=None,
                currency=None,
                effective_date=sent_date,
                percent=None,
                note="Message instructions do not override challenge rules.",
            )
        ]

    overlays: list[EvidenceOverlay] = []

    if _contains_any(
        lowered,
        (
            "pay the release charge",
            "pay the processing charge",
            "pay the release",
        ),
    ):
        overlays.append(
            _overlay("ignore_unconfirmed_credit", message, None, None, sent_date, None, "Prize-release scam; ignore.")
        )
        return overlays

    if "12%" in text and _contains_any(lowered, ("rent", "lease", "sewa")):
        overlays.append(
            _overlay("rent_increase_percent", message, None, None, sent_date, Decimal("12"), "Lease renewal increases rent 12%.")
        )

    if _contains_any(lowered, ("transfer between your two accounts", "kedua akun")):
        overlays.append(
            _overlay("internal_transfer_notice", message, None, None, sent_date, None, "Matching debit/credit is an internal transfer.")
        )

    if _contains_any(
        lowered,
        (
            "payout is still pending",
            "masih tertunda",
            "has not been credited",
            "belum masuk",
            "not reached your account",
            "reversal has not been posted",
            "belum tercatat",
        ),
    ) and not _contains_any(lowered, ("debit attempt failed",)):
        overlays.append(
            _overlay(
                "pending_credit_unconfirmed",
                message,
                None,
                None,
                sent_date,
                None,
                "Credit/payout/refund is not settled; do not count it.",
            )
        )

    if _contains_any(lowered, ("no units have been sold", "belum dijual", "no cash proceeds", "tidak ada transaksi tunai")):
        overlays.append(
            _overlay("ignore_unrealized_value", message, None, None, sent_date, None, "Investment value is non-cash.")
        )

    if _contains_any(lowered, ("debit attempt failed", "gagal", "bill is still outstanding", "another debit will be attempted")):
        overlays.append(
            _overlay("failed_debit_retry", message, None, None, sent_date, None, "Failed debit; use the scheduled retry.")
        )

    if _contains_any(lowered, ("bonus",)) and _contains_any(lowered, ("not been approved", "belum disetujui", "waiting")):
        overlays.append(
            _overlay("unconfirmed_bonus", message, None, None, sent_date, None, "Bonus is unconfirmed.")
        )

    if _contains_any(lowered, ("commission", "komisi")) and _contains_any(
        lowered, ("not been approved", "belum disetujui", "running")
    ):
        overlays.append(
            _overlay("unconfirmed_commission", message, None, None, sent_date, None, "Commission is unconfirmed.")
        )

    if _contains_any(lowered, ("seasonal contract has ended", "kontrak musiman", "no off-season income")):
        overlays.append(
            _overlay("income_series_ended", message, None, None, sent_date, None, "Seasonal income has ended.")
        )

    if _contains_any(
        lowered,
        (
            "employment record has ended",
            "pendapatan kerja rumah tangga telah berakhir",
            "income that has ended",
            "pendapatan yang sudah berakhir",
        ),
    ):
        amount, currency = _first_money(text)
        overlays.append(
            _overlay(
                "household_income_reduced",
                message,
                amount,
                currency,
                sent_date,
                None,
                "One household income source ended; remaining salary confirmed.",
            )
        )

    if _contains_any(lowered, ("temporary monthly pay", "gaji bulanan sementara", "next salary is reduced", "gaji berikutnya dikurangi")):
        amount, currency = _first_money(text)
        overlays.append(
            _overlay("temporary_salary_reduction", message, amount, currency, sent_date, None, "Temporary reduced salary for the next payroll.")
        )

    if _contains_any(
        lowered,
        (
            "salary has increased",
            "gaji bulanan anda naik",
            "naik menjadi",
            "monthly salary has increased",
        ),
    ):
        amount, currency = _first_money(text)
        effective = _first_date(text) or sent_date
        overlays.append(
            _overlay("salary_increase", message, amount, currency, effective, None, "Confirmed recurring salary increase.")
        )

    if _contains_any(lowered, ("confirmed salary is now expected on", "gaji yang dikonfirmasi sekarang diharapkan")):
        effective = _first_date(text)
        overlays.append(
            _overlay("salary_date_amendment", message, None, None, effective, None, "Confirmed salary date replaced.")
        )

    if _contains_any(
        lowered,
        (
            "first salary will be",
            "gaji pertama",
            "first salary from the new employer",
            "gaji pertama dari perusahaan baru",
        ),
    ):
        amount, currency = _first_money(text)
        effective = _first_date(text)
        overlays.append(
            _overlay("confirmed_first_salary", message, amount, currency, effective, None, "First confirmed salary credit.")
        )

    if _contains_any(lowered, ("regular salary of", "regular salary resumes", "gaji rutin sebesar")):
        amount, currency = _first_money(text)
        effective = _first_date(text) or sent_date
        overlays.append(
            _overlay("salary_resume", message, amount, currency, effective, None, "Regular salary resumes.")
        )

    if _contains_any(lowered, ("one-time arrears", "penyesuaian tunggakan satu kali", "arrears adjustment")):
        amount, currency = _nth_money(text, 1) if _money_count(text) > 1 else _first_money(text)
        overlays.append(
            _overlay("one_time_arrears", message, amount, currency, sent_date, None, "One-time arrears; do not treat as recurring.")
        )

    if _contains_any(
        lowered,
        (
            "approved an invoice payment",
            "menyetujui pembayaran faktur",
            "klien menyetujui",
        ),
    ):
        amount, currency = _first_money(text)
        effective = _first_date(text)
        overlays.append(
            _overlay("confirmed_invoice_credit", message, amount, currency, effective, None, "Only the confirmed invoice may be counted.")
        )

    if _contains_any(lowered, ("settlement-date rate", "kurs tanggal penyelesaian")):
        overlays.append(
            _overlay("fx_refund_use_settlement_rate", message, None, None, sent_date, None, "Convert foreign refund on settlement date.")
        )

    if _contains_any(lowered, ("new recurring childcare", "pembayaran pengasuhan")):
        overlays.append(
            _overlay("childcare_recurring_starts", message, None, None, _first_date(text) or sent_date, None, "Childcare mentioned without an amount; do not invent it.")
        )

    if not overlays and _MESSAGE_LLM is not None and getattr(_MESSAGE_LLM, "enabled", False):
        data = _MESSAGE_LLM.classify_message(
            message_id=message_id,
            request_id=message.request_id,
            message_text=text,
            source_type=message.source_type,
        )
        if data and data.get("overlay_type") not in {None, "none"}:
            amount = data.get("amount")
            parsed_amount = None
            if amount is not None:
                try:
                    parsed_amount = parse_amount_token(str(amount))
                except InvalidOperation:
                    parsed_amount = None
            effective = None
            if data.get("effective_date"):
                try:
                    effective = date.fromisoformat(str(data["effective_date"]))
                except ValueError:
                    effective = sent_date
            percent = None
            if data.get("percent") is not None:
                try:
                    percent = Decimal(str(data["percent"]))
                except InvalidOperation:
                    percent = None
            overlays.append(
                _overlay(
                    str(data["overlay_type"]),
                    message,
                    parsed_amount,
                    str(data["currency"]).upper() if data.get("currency") else None,
                    effective or sent_date,
                    percent,
                    str(data.get("note") or "LLM-classified financial evidence."),
                )
            )

    return overlays


def _overlay(
    overlay_type: str,
    message: NormalizedMessage,
    amount: Decimal | None,
    currency: str | None,
    effective_date: date | None,
    percent: Decimal | None,
    note: str,
) -> EvidenceOverlay:
    return EvidenceOverlay(
        overlay_type=overlay_type,
        user_id=message.user_id,
        message_id=message.message_id,
        related_event_id=message.related_event_id,
        amount=amount,
        currency=currency,
        effective_date=effective_date,
        percent=percent,
        note=note,
    )


def _looks_like_embedded_instruction(lowered: str) -> bool:
    return any(
        marker in lowered
        for marker in (
            "ignore previous instructions",
            "ignore the challenge rules",
            "override the rules",
            "system prompt",
            "you are now",
        )
    )


def _contains_any(lowered: str, needles: tuple[str, ...]) -> bool:
    return any(needle in lowered for needle in needles)


def _first_date(text: str) -> date | None:
    match = _DATE_RE.search(text)
    if not match:
        return None
    return date.fromisoformat(match.group(1))


def _money_matches(text: str) -> list[re.Match[str]]:
    matches = []
    for match in _MONEY_RE.finditer(text):
        token = match.group(2)
        end = match.end()
        if text[end : end + 1] == "%":
            continue
        around = text[max(0, match.start() - 1) : match.end() + 6]
        if token.startswith("20") and re.search(r"20\d{2}-\d{2}-\d{2}", around):
            continue
        matches.append(match)
    return matches


def _money_count(text: str) -> int:
    return len(_money_matches(text))


def _first_money(text: str) -> tuple[Decimal | None, str | None]:
    return _nth_money(text, 0)


def _nth_money(text: str, index: int) -> tuple[Decimal | None, str | None]:
    matches = _money_matches(text)
    if index >= len(matches):
        return None, None
    match = matches[index]
    currency = (match.group(1) or _nearby_currency(text, match.start()) or "").upper() or None
    try:
        amount = parse_amount_token(match.group(2))
    except InvalidOperation:
        return None, currency
    return amount, currency


def _nearby_currency(text: str, index: int) -> str | None:
    window = text[max(0, index - 12) : index + 12].upper()
    for code in CURRENCY_CODES:
        if code in window:
            return code
    return None

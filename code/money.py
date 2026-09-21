from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP


def format_amount(amount: Decimal) -> str:
    quantized = amount.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    text = format(quantized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def format_amount_for_option(amount: Decimal, option_amount: str) -> str:
    if Decimal(option_amount) == amount:
        return option_amount
    return format_amount(amount)


def format_grouped(amount: Decimal, currency: str) -> str:
    formatted = format_amount(amount)
    if "." in formatted:
        whole, frac = formatted.split(".", 1)
    else:
        whole, frac = formatted, ""
    sign = ""
    if whole.startswith("-"):
        sign = "-"
        whole = whole[1:]
    grouped = ",".join([whole[max(i - 3, 0) : i] for i in range(len(whole), 0, -3)][::-1]) or "0"
    body = f"{sign}{grouped}.{frac}" if frac else f"{sign}{grouped}"
    return f"{currency} {body}"

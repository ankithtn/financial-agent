from __future__ import annotations

from datetime import date
from decimal import Decimal

from normalization import NormalizedExchangeRate, NormalizationError


def convert_to_home(
    amount: Decimal,
    from_currency: str,
    to_currency: str,
    rate_date: date,
    rates_by_key: dict[tuple[date, str, str], NormalizedExchangeRate],
    *,
    field_name: str,
) -> Decimal:
    if from_currency == to_currency:
        return amount
    rate = rates_by_key.get((rate_date, from_currency, to_currency))
    if rate is None:
        raise NormalizationError(
            f"Missing exchange rate for {field_name}: {rate_date} {from_currency}->{to_currency}"
        )
    return amount * rate.rate

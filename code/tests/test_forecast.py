from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from forecast import Forecast, ForecastFlow, max_safe_payment, simulate
from money import format_amount


def _forecast(opening: Decimal, minimum: Decimal, flows: tuple[ForecastFlow, ...] = ()) -> Forecast:
    return Forecast(
        request_date=date(2026, 1, 1),
        horizon=date(2026, 4, 1),
        opening_balance=opening,
        minimum_balance_to_keep=minimum,
        flows=flows,
        series=(),
    )


class ForecastTests(unittest.TestCase):
    def test_payment_today_is_capped_by_minimum_balance(self) -> None:
        forecast = _forecast(Decimal("1000"), Decimal("400"))
        self.assertEqual(max_safe_payment(forecast, Decimal("1000")), Decimal("600.00"))

    def test_future_debit_reduces_safe_amount(self) -> None:
        flow = ForecastFlow(
            on=date(2026, 1, 15),
            signed_amount=Decimal("-200"),
            category="rent",
            description="rent",
            series_id=None,
            change_event_id=None,
            flexibility="fixed",
            minimum_allowed=None,
            source="test",
        )
        forecast = _forecast(Decimal("1000"), Decimal("400"), (flow,))
        self.assertEqual(max_safe_payment(forecast, Decimal("1000")), Decimal("400.00"))

    def test_pending_credit_not_in_forecast_cannot_increase_safe_amount(self) -> None:
        forecast = _forecast(Decimal("500"), Decimal("400"))
        safe, lowest = simulate(forecast, ((date(2026, 1, 1), Decimal("100")),))
        self.assertTrue(safe)
        self.assertEqual(lowest, Decimal("400"))

    def test_format_amount_strips_trailing_zeros(self) -> None:
        self.assertEqual(format_amount(Decimal("620.40")), "620.4")
        self.assertEqual(format_amount(Decimal("25256")), "25256")


if __name__ == "__main__":
    unittest.main()

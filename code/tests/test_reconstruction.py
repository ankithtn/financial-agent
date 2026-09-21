from __future__ import annotations

import sys
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_loader import load_dataset
from evidence import EXTRACTED_IMAGE_AMOUNTS
from normalization import (
    NormalizedExchangeRate,
    NormalizedFinancialEvent,
    NormalizedFinancialProfile,
    NormalizedImageReference,
    normalize_dataset,
)
from reconstruction import (
    CASH_IGNORE,
    CASH_INCLUDE,
    CASH_RESERVE,
    CADENCE_RECURRING,
    reconstruct_dataset,
    reconstruct_user_state,
    reconstruction_summary,
)


def _profile(**overrides) -> NormalizedFinancialProfile:
    values = dict(
        user_id="user_x",
        home_currency="INR",
        current_available_balance=Decimal("10000"),
        minimum_balance_to_keep=Decimal("1000"),
        financial_priorities=("education",),
        expense_categories_to_protect=("rent",),
        expense_categories_user_is_willing_to_reduce=("dining",),
        expense_categories_user_is_willing_to_stop=("streaming",),
        payment_methods_user_will_consider=("full_payment",),
        max_installment_months=None,
    )
    values.update(overrides)
    return NormalizedFinancialProfile(**values)


def _event(**overrides) -> NormalizedFinancialEvent:
    values = dict(
        event_id="event_x",
        user_id="user_x",
        event_type="expense",
        description="Test expense",
        category="shopping",
        direction="debit",
        amount=Decimal("100"),
        currency="INR",
        event_date=date(2026, 1, 1),
        settlement_date=date(2026, 1, 1),
        status="settled",
        linked_event_id=None,
        flexibility="fixed",
        minimum_allowed_amount=None,
    )
    values.update(overrides)
    return NormalizedFinancialEvent(**values)


class ReconstructionTests(unittest.TestCase):
    def test_cancelled_parent_ignored_settled_child_included(self) -> None:
        parent = _event(event_id="event_c", status="cancelled", description="Cancelled card")
        child = _event(
            event_id="event_s",
            status="settled",
            linked_event_id="event_c",
            event_date=date(2026, 1, 3),
            settlement_date=date(2026, 1, 3),
            description="Settled card purchase",
        )
        state = reconstruct_user_state(_profile(), [parent, child], exchange_rates_by_key={})
        self.assertEqual(state.events_by_id["event_c"].cash_role, CASH_IGNORE)
        self.assertEqual(state.events_by_id["event_s"].cash_role, CASH_INCLUDE)
        self.assertIn("settled_amendment_after_cancellation", "|".join(state.events_by_id["event_s"].provenance))

    def test_failed_debit_ignored_scheduled_retry_included(self) -> None:
        failed = _event(
            event_id="event_f",
            event_type="debt_payment",
            category="debt_repayment",
            status="failed",
            description="Failed bill payment attempt",
        )
        retry = _event(
            event_id="event_r",
            event_type="debt_payment",
            category="debt_repayment",
            status="scheduled",
            linked_event_id="event_f",
            event_date=date(2026, 1, 3),
            settlement_date=date(2026, 1, 3),
            description="Scheduled bill payment retry",
        )
        state = reconstruct_user_state(_profile(), [failed, retry], exchange_rates_by_key={})
        self.assertEqual(state.events_by_id["event_f"].cash_role, CASH_IGNORE)
        self.assertEqual(state.events_by_id["event_r"].cash_role, CASH_INCLUDE)

    def test_pending_refund_ignored_pending_debit_reserved(self) -> None:
        refund = _event(
            event_id="event_refund",
            event_type="refund",
            direction="credit",
            status="pending",
            description="Pending merchant refund",
        )
        duplicate = _event(
            event_id="event_dup",
            status="pending",
            description="Possible duplicate card charge",
            settlement_date=date(2026, 1, 10),
        )
        state = reconstruct_user_state(_profile(), [refund, duplicate], exchange_rates_by_key={})
        self.assertEqual(state.events_by_id["event_refund"].cash_role, CASH_IGNORE)
        self.assertEqual(state.events_by_id["event_dup"].cash_role, CASH_RESERVE)

    def test_unrealized_investment_is_not_cash(self) -> None:
        valuation = _event(
            event_id="event_u",
            event_type="investment_valuation",
            direction="non_cash",
            status="unrealized",
            category="investment",
            settlement_date=None,
            description="Current portfolio valuation",
        )
        state = reconstruct_user_state(_profile(), [valuation], exchange_rates_by_key={})
        self.assertEqual(state.events_by_id["event_u"].cash_role, CASH_IGNORE)

    def test_fx_conversion_uses_settlement_date_rate(self) -> None:
        event = _event(amount=Decimal("10"), currency="USD", settlement_date=date(2026, 1, 15))
        rate = NormalizedExchangeRate(
            rate_date=date(2026, 1, 15),
            from_currency="USD",
            to_currency="INR",
            rate=Decimal("80"),
        )
        state = reconstruct_user_state(
            _profile(home_currency="INR"),
            [event],
            exchange_rates_by_key={(rate.rate_date, rate.from_currency, rate.to_currency): rate},
        )
        self.assertEqual(state.events_by_id["event_x"].amount_home, Decimal("800"))

    def test_blank_amount_is_filled_from_image(self) -> None:
        event = _event(event_id="event_blank", amount=None)
        image = NormalizedImageReference(
            image_id="image_07",
            user_id="user_x",
            request_id=None,
            related_event_id="event_blank",
            path=CODE_DIR.parent / "dataset" / "media" / "images" / "image_07.png",
        )
        state = reconstruct_user_state(
            _profile(),
            [event],
            exchange_rates_by_key={},
            images_by_event_id={"event_blank": (image,)},
        )
        reconstructed = state.events_by_id["event_blank"]
        self.assertEqual(reconstructed.amount_home, EXTRACTED_IMAGE_AMOUNTS["image_07"])
        self.assertEqual(reconstructed.amount_source, "image")

    def test_monthly_rent_is_recurring_one_off_purchase_is_not(self) -> None:
        rent = [
            _event(
                event_id=f"rent_{index}",
                description="Apartment rent transfer",
                category="rent",
                amount=Decimal("5000"),
                event_date=date(2026, index, 2),
                settlement_date=date(2026, index, 2),
            )
            for index in range(1, 5)
        ]
        purchase = _event(event_id="bag", description="Tote bag order", category="shopping", amount=Decimal("2298"))
        state = reconstruct_user_state(_profile(), [*rent, purchase], exchange_rates_by_key={})
        self.assertEqual(state.events_by_id["rent_1"].cadence, CADENCE_RECURRING)
        self.assertEqual(state.events_by_id["rent_1"].recurrence_days, 30)
        self.assertEqual(state.events_by_id["bag"].cadence, "one_time")
        self.assertEqual(len(state.series), 1)

    def test_salary_history_and_next_confirmed_salary_share_a_series(self) -> None:
        payroll = [
            _event(
                event_id=f"pay_{index}",
                event_type="income",
                direction="credit",
                category="salary",
                description="Payroll credit",
                amount=Decimal("2000"),
                event_date=date(2026, index, 15),
                settlement_date=date(2026, index, 15),
            )
            for index in range(1, 4)
        ]
        nxt = _event(
            event_id="next_pay",
            event_type="income",
            direction="credit",
            category="salary",
            description="Next confirmed salary",
            status="scheduled",
            amount=Decimal("2000"),
            event_date=date(2026, 4, 15),
            settlement_date=date(2026, 4, 15),
        )
        state = reconstruct_user_state(_profile(), [*payroll, nxt], exchange_rates_by_key={})
        self.assertEqual(state.events_by_id["next_pay"].cash_role, CASH_INCLUDE)
        self.assertEqual(state.events_by_id["pay_1"].series_id, state.events_by_id["next_pay"].series_id)
        self.assertEqual(state.events_by_id["next_pay"].cadence, CADENCE_RECURRING)

    def test_full_dataset_reconstruction_fills_images_and_ignores_cancelled(self) -> None:
        normalized = normalize_dataset(load_dataset(CODE_DIR.parent))
        reconstructed = reconstruct_dataset(normalized)
        summary = reconstruction_summary(reconstructed)

        self.assertEqual(summary["reconstructed_users"], 275)
        self.assertEqual(summary["reconstructed_events"], 25342)
        self.assertEqual(summary["amounts_from_images"], 16)
        self.assertEqual(summary["unresolved_amounts"], 0)

        cancelled = [event for event in reconstructed.events_by_id.values() if event.status == "cancelled"]
        self.assertTrue(cancelled)
        self.assertTrue(all(event.cash_role == CASH_IGNORE for event in cancelled))

        event_253 = reconstructed.events_by_id["event_253"]
        self.assertEqual(event_253.amount_source, "image")
        self.assertEqual(event_253.original_amount, Decimal("4365000"))

        event_5168 = reconstructed.events_by_id["event_5168"]
        event_5169 = reconstructed.events_by_id["event_5169"]
        self.assertEqual(event_5168.cash_role, CASH_IGNORE)
        self.assertEqual(event_5169.cash_role, CASH_INCLUDE)

        user_01 = reconstructed.users["user_01"]
        rent_series = [series for series in user_01.series if series.category == "rent"]
        self.assertEqual(len(rent_series), 1)
        self.assertGreaterEqual(rent_series[0].occurrence_count, 3)
        self.assertTrue(any(overlay.overlay_type == "salary_increase" for overlay in reconstructed.users["user_02"].overlays))


if __name__ == "__main__":
    unittest.main()

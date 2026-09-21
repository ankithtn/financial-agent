from __future__ import annotations

import sys
import unittest
from datetime import date, timezone
from decimal import Decimal
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from data_loader import load_dataset
from normalization import normalize_dataset


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = CODE_DIR.parent
        cls.dataset = normalize_dataset(load_dataset(cls.repo_root))

    def test_request_fields_are_typed_and_ids_preserved(self) -> None:
        request = self.dataset.requests_by_request_id["request_26"]

        self.assertEqual(request.request_id, "request_26")
        self.assertEqual(request.user_id, "user_26")
        self.assertEqual(request.request_date, date(2025, 8, 3))
        self.assertEqual(request.desired_completion_date, date(2025, 10, 7))
        self.assertEqual(request.requested_amount, Decimal("15656000"))
        self.assertFalse(request.allows_partial_payment)

    def test_profile_splits_preferences_and_optional_installment_limit(self) -> None:
        profile = self.dataset.profiles_by_user_id["user_02"]
        profile_without_installments = self.dataset.profiles_by_user_id["user_01"]

        self.assertEqual(profile.home_currency, "IDR")
        self.assertEqual(profile.current_available_balance, Decimal("60383889.2"))
        self.assertIn("installments", profile.payment_methods_user_will_consider)
        self.assertEqual(profile.max_installment_months, 7)
        self.assertIsNone(profile_without_installments.max_installment_months)

    def test_blank_event_amounts_are_preserved_for_image_extraction(self) -> None:
        event = self.dataset.events_by_event_id["event_253"]
        image = self.dataset.images_by_event_id["event_253"][0]

        self.assertIsNone(event.amount)
        self.assertEqual(event.currency, "IDR")
        self.assertEqual(event.settlement_date, date(2019, 8, 31))
        self.assertEqual(image.image_id, "image_01")
        self.assertTrue(image.path.exists())

    def test_unrealized_non_cash_event_can_have_no_settlement_date(self) -> None:
        event = self.dataset.events_by_event_id["event_1856"]

        self.assertEqual(event.status, "unrealized")
        self.assertEqual(event.direction, "non_cash")
        self.assertIsNone(event.settlement_date)

    def test_exchange_rate_and_payment_option_are_typed(self) -> None:
        rate = self.dataset.exchange_rates_by_key[(date(2023, 10, 15), "USD", "IDR")]
        option = self.dataset.payment_options_by_request_id["request_02"][0]

        self.assertEqual(rate.rate, Decimal("15833.33"))
        self.assertEqual(option.payment_option_id, "payment_option_05")
        self.assertEqual(option.payment_amount, Decimal("15952906.67"))
        self.assertEqual(option.number_of_payments, 3)
        self.assertEqual(option.payment_frequency_days, 30)

    def test_message_datetime_is_utc_and_optional_ids_are_none(self) -> None:
        message = self.dataset.messages_by_user_id["user_02"][0]

        self.assertIsNone(message.request_id)
        self.assertIsNone(message.related_event_id)
        self.assertEqual(message.sent_at.tzinfo, timezone.utc)


if __name__ == "__main__":
    unittest.main()

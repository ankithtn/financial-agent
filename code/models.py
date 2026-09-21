from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Request:
    request_id: str
    user_id: str
    request_date: str
    request_type: str
    requested_amount: str
    desired_completion_date: str
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class SampleRequest:
    request_id: str
    user_id: str
    request_date: str
    request_type: str
    requested_amount: str
    desired_completion_date: str
    allows_partial_payment: bool
    request_text: str
    amount_safe_to_pay: str
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: str
    minimum_balance_to_keep: str
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: str


@dataclass(frozen=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: str
    currency: str
    event_date: str
    settlement_date: str
    status: str
    linked_event_id: str
    flexibility: str
    minimum_allowed_amount: str


@dataclass(frozen=True)
class ExchangeRate:
    rate_date: str
    from_currency: str
    to_currency: str
    rate: str


@dataclass(frozen=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: str
    number_of_payments: str
    first_payment_date: str
    payment_frequency_days: str
    financing_fee: str
    total_payable_amount: str


@dataclass(frozen=True)
class Message:
    message_id: str
    user_id: str
    request_id: str
    related_event_id: str
    sent_at: str
    source_type: str
    message_text: str


@dataclass(frozen=True)
class ImageReference:
    image_id: str
    user_id: str
    request_id: str
    related_event_id: str

    @property
    def relative_path(self) -> Path:
        return Path("dataset") / "media" / "images" / f"{self.image_id}.png"


@dataclass(frozen=True)
class OutputTemplateRow:
    request_id: str
    amount_safe_to_pay: str
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class DatasetBundle:
    repo_root: Path
    requests: tuple[Request, ...]
    sample_requests: tuple[SampleRequest, ...]
    financial_profiles: tuple[FinancialProfile, ...]
    financial_events: tuple[FinancialEvent, ...]
    exchange_rates: tuple[ExchangeRate, ...]
    payment_options: tuple[PaymentOption, ...]
    messages: tuple[Message, ...]
    images: tuple[ImageReference, ...]
    output_template: tuple[OutputTemplateRow, ...]
    profiles_by_user_id: dict[str, FinancialProfile]
    requests_by_request_id: dict[str, Request]
    events_by_event_id: dict[str, FinancialEvent]
    events_by_user_id: dict[str, tuple[FinancialEvent, ...]]
    payment_options_by_request_id: dict[str, tuple[PaymentOption, ...]]
    messages_by_user_id: dict[str, tuple[Message, ...]]
    messages_by_request_id: dict[str, tuple[Message, ...]]
    messages_by_event_id: dict[str, tuple[Message, ...]]
    images_by_request_id: dict[str, tuple[ImageReference, ...]]
    images_by_event_id: dict[str, tuple[ImageReference, ...]]
    exchange_rates_by_key: dict[tuple[str, str, str], ExchangeRate]


def split_pipe(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(part for part in value.split("|") if part)

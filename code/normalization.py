from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from data_loader import DatasetValidationError
from models import DatasetBundle


SUPPORTED_CURRENCIES = {"EUR", "IDR", "INR", "USD", "ZAR"}
REQUEST_TYPES = {
    "purchase",
    "travel",
    "education",
    "family_transfer",
    "debt_repayment",
    "investment",
    "housing",
    "emergency_expense",
    "other",
}
EVENT_TYPES = {
    "debt_payment",
    "expense",
    "income",
    "investment_purchase",
    "investment_sale",
    "investment_valuation",
    "refund",
    "subscription",
}
DIRECTIONS = {"credit", "debit", "non_cash"}
EVENT_STATUSES = {"cancelled", "failed", "pending", "scheduled", "settled", "unrealized"}
FLEXIBILITY_VALUES = {"fixed", "reducible", "reducible_or_stoppable", "stoppable"}
PAYMENT_METHOD_VALUES = {"full_payment", "installments"}


class NormalizationError(ValueError):
    pass


@dataclass(frozen=True)
class NormalizedRequest:
    request_id: str
    user_id: str
    request_date: date
    request_type: str
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True)
class NormalizedSampleRequest(NormalizedRequest):
    amount_safe_to_pay: Decimal
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class NormalizedFinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[str, ...]
    max_installment_months: int | None


@dataclass(frozen=True)
class NormalizedFinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Decimal | None
    currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str | None
    flexibility: str
    minimum_allowed_amount: Decimal | None


@dataclass(frozen=True)
class NormalizedExchangeRate:
    rate_date: date
    from_currency: str
    to_currency: str
    rate: Decimal


@dataclass(frozen=True)
class NormalizedPaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True)
class NormalizedMessage:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: datetime
    source_type: str
    message_text: str


@dataclass(frozen=True)
class NormalizedImageReference:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    path: Path


@dataclass(frozen=True)
class NormalizedDataset:
    repo_root: Path
    requests: tuple[NormalizedRequest, ...]
    sample_requests: tuple[NormalizedSampleRequest, ...]
    profiles: tuple[NormalizedFinancialProfile, ...]
    events: tuple[NormalizedFinancialEvent, ...]
    exchange_rates: tuple[NormalizedExchangeRate, ...]
    payment_options: tuple[NormalizedPaymentOption, ...]
    messages: tuple[NormalizedMessage, ...]
    images: tuple[NormalizedImageReference, ...]
    profiles_by_user_id: dict[str, NormalizedFinancialProfile]
    requests_by_request_id: dict[str, NormalizedRequest]
    sample_requests_by_request_id: dict[str, NormalizedSampleRequest]
    events_by_event_id: dict[str, NormalizedFinancialEvent]
    events_by_user_id: dict[str, tuple[NormalizedFinancialEvent, ...]]
    exchange_rates_by_key: dict[tuple[date, str, str], NormalizedExchangeRate]
    payment_options_by_request_id: dict[str, tuple[NormalizedPaymentOption, ...]]
    messages_by_user_id: dict[str, tuple[NormalizedMessage, ...]]
    messages_by_request_id: dict[str, tuple[NormalizedMessage, ...]]
    messages_by_event_id: dict[str, tuple[NormalizedMessage, ...]]
    images_by_request_id: dict[str, tuple[NormalizedImageReference, ...]]
    images_by_event_id: dict[str, tuple[NormalizedImageReference, ...]]


def normalize_dataset(bundle: DatasetBundle) -> NormalizedDataset:
    requests = tuple(
        NormalizedRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            request_date=_date(request.request_date, f"{request.request_id}.request_date"),
            request_type=_choice(request.request_type, REQUEST_TYPES, f"{request.request_id}.request_type"),
            requested_amount=_decimal(request.requested_amount, f"{request.request_id}.requested_amount"),
            desired_completion_date=_date(
                request.desired_completion_date,
                f"{request.request_id}.desired_completion_date",
            ),
            allows_partial_payment=request.allows_partial_payment,
            request_text=request.request_text,
        )
        for request in bundle.requests
    )

    sample_requests = tuple(
        NormalizedSampleRequest(
            request_id=request.request_id,
            user_id=request.user_id,
            request_date=_date(request.request_date, f"{request.request_id}.request_date"),
            request_type=_choice(request.request_type, REQUEST_TYPES, f"{request.request_id}.request_type"),
            requested_amount=_decimal(request.requested_amount, f"{request.request_id}.requested_amount"),
            desired_completion_date=_date(
                request.desired_completion_date,
                f"{request.request_id}.desired_completion_date",
            ),
            allows_partial_payment=request.allows_partial_payment,
            request_text=request.request_text,
            amount_safe_to_pay=_decimal(request.amount_safe_to_pay, f"{request.request_id}.amount_safe_to_pay"),
            affordability_status=request.affordability_status,
            recommended_payment_method=request.recommended_payment_method,
            payment_plan=request.payment_plan,
            earliest_date_for_full_payment=_optional_date(
                request.earliest_date_for_full_payment,
                f"{request.request_id}.earliest_date_for_full_payment",
            ),
            spending_changes_needed=request.spending_changes_needed,
            decision_explanation=request.decision_explanation,
        )
        for request in bundle.sample_requests
    )

    profiles = tuple(
        NormalizedFinancialProfile(
            user_id=profile.user_id,
            home_currency=_currency(profile.home_currency, f"{profile.user_id}.home_currency"),
            current_available_balance=_decimal(
                profile.current_available_balance,
                f"{profile.user_id}.current_available_balance",
            ),
            minimum_balance_to_keep=_decimal(
                profile.minimum_balance_to_keep,
                f"{profile.user_id}.minimum_balance_to_keep",
            ),
            financial_priorities=profile.financial_priorities,
            expense_categories_to_protect=profile.expense_categories_to_protect,
            expense_categories_user_is_willing_to_reduce=profile.expense_categories_user_is_willing_to_reduce,
            expense_categories_user_is_willing_to_stop=profile.expense_categories_user_is_willing_to_stop,
            payment_methods_user_will_consider=profile.payment_methods_user_will_consider,
            max_installment_months=_optional_int(
                profile.max_installment_months,
                f"{profile.user_id}.max_installment_months",
            ),
        )
        for profile in bundle.financial_profiles
    )

    events = tuple(
        NormalizedFinancialEvent(
            event_id=event.event_id,
            user_id=event.user_id,
            event_type=_choice(event.event_type, EVENT_TYPES, f"{event.event_id}.event_type"),
            description=event.description,
            category=event.category,
            direction=_choice(event.direction, DIRECTIONS, f"{event.event_id}.direction"),
            amount=_optional_decimal(event.amount, f"{event.event_id}.amount"),
            currency=_currency(event.currency, f"{event.event_id}.currency"),
            event_date=_date(event.event_date, f"{event.event_id}.event_date"),
            settlement_date=_optional_date(event.settlement_date, f"{event.event_id}.settlement_date"),
            status=_choice(event.status, EVENT_STATUSES, f"{event.event_id}.status"),
            linked_event_id=event.linked_event_id or None,
            flexibility=_choice(event.flexibility, FLEXIBILITY_VALUES, f"{event.event_id}.flexibility"),
            minimum_allowed_amount=_optional_decimal(
                event.minimum_allowed_amount,
                f"{event.event_id}.minimum_allowed_amount",
            ),
        )
        for event in bundle.financial_events
    )

    exchange_rates = tuple(
        NormalizedExchangeRate(
            rate_date=_date(rate.rate_date, f"{rate.from_currency}->{rate.to_currency}.rate_date"),
            from_currency=_currency(rate.from_currency, "exchange_rate.from_currency"),
            to_currency=_currency(rate.to_currency, "exchange_rate.to_currency"),
            rate=_decimal(rate.rate, f"{rate.rate_date}.{rate.from_currency}->{rate.to_currency}.rate"),
        )
        for rate in bundle.exchange_rates
    )

    payment_options = tuple(
        NormalizedPaymentOption(
            payment_option_id=option.payment_option_id,
            request_id=option.request_id,
            payment_method=_choice(
                option.payment_method,
                PAYMENT_METHOD_VALUES,
                f"{option.payment_option_id}.payment_method",
            ),
            payment_amount=_decimal(option.payment_amount, f"{option.payment_option_id}.payment_amount"),
            number_of_payments=_int(option.number_of_payments, f"{option.payment_option_id}.number_of_payments"),
            first_payment_date=_date(option.first_payment_date, f"{option.payment_option_id}.first_payment_date"),
            payment_frequency_days=_optional_int(
                option.payment_frequency_days,
                f"{option.payment_option_id}.payment_frequency_days",
            ),
            financing_fee=_decimal(option.financing_fee, f"{option.payment_option_id}.financing_fee"),
            total_payable_amount=_decimal(
                option.total_payable_amount,
                f"{option.payment_option_id}.total_payable_amount",
            ),
        )
        for option in bundle.payment_options
    )

    messages = tuple(
        NormalizedMessage(
            message_id=message.message_id,
            user_id=message.user_id,
            request_id=message.request_id or None,
            related_event_id=message.related_event_id or None,
            sent_at=_datetime_utc(message.sent_at, f"{message.message_id}.sent_at"),
            source_type=message.source_type,
            message_text=message.message_text,
        )
        for message in bundle.messages
    )

    images = tuple(
        NormalizedImageReference(
            image_id=image.image_id,
            user_id=image.user_id,
            request_id=image.request_id or None,
            related_event_id=image.related_event_id or None,
            path=bundle.repo_root / image.relative_path,
        )
        for image in bundle.images
    )

    normalized = NormalizedDataset(
        repo_root=bundle.repo_root,
        requests=requests,
        sample_requests=sample_requests,
        profiles=profiles,
        events=events,
        exchange_rates=exchange_rates,
        payment_options=payment_options,
        messages=messages,
        images=images,
        profiles_by_user_id={profile.user_id: profile for profile in profiles},
        requests_by_request_id={request.request_id: request for request in requests},
        sample_requests_by_request_id={request.request_id: request for request in sample_requests},
        events_by_event_id={event.event_id: event for event in events},
        events_by_user_id=_group_by(events, lambda event: event.user_id),
        exchange_rates_by_key={(rate.rate_date, rate.from_currency, rate.to_currency): rate for rate in exchange_rates},
        payment_options_by_request_id=_group_by(payment_options, lambda option: option.request_id),
        messages_by_user_id=_group_by(messages, lambda message: message.user_id),
        messages_by_request_id=_group_nonblank(messages, lambda message: message.request_id),
        messages_by_event_id=_group_nonblank(messages, lambda message: message.related_event_id),
        images_by_request_id=_group_nonblank(images, lambda image: image.request_id),
        images_by_event_id=_group_nonblank(images, lambda image: image.related_event_id),
    )
    _validate_normalized_dataset(normalized)
    return normalized


def normalized_summary(dataset: NormalizedDataset) -> dict[str, int]:
    return {
        "normalized_requests": len(dataset.requests),
        "normalized_sample_requests": len(dataset.sample_requests),
        "normalized_profiles": len(dataset.profiles),
        "normalized_events": len(dataset.events),
        "normalized_exchange_rates": len(dataset.exchange_rates),
        "normalized_payment_options": len(dataset.payment_options),
        "normalized_messages": len(dataset.messages),
        "normalized_images": len(dataset.images),
        "events_missing_amount": sum(1 for event in dataset.events if event.amount is None),
        "events_missing_settlement_date": sum(1 for event in dataset.events if event.settlement_date is None),
    }


def _validate_normalized_dataset(dataset: NormalizedDataset) -> None:
    profile_user_ids = set(dataset.profiles_by_user_id)
    request_ids = set(dataset.requests_by_request_id) | set(dataset.sample_requests_by_request_id)
    event_ids = set(dataset.events_by_event_id)

    for request in (*dataset.requests, *dataset.sample_requests):
        _require(request.user_id in profile_user_ids, f"{request.request_id}: normalized request has unknown user.")
        _require(request.requested_amount >= 0, f"{request.request_id}: requested_amount cannot be negative.")
        _require(
            request.desired_completion_date >= request.request_date,
            f"{request.request_id}: desired_completion_date is before request_date.",
        )

    for profile in dataset.profiles:
        for method in profile.payment_methods_user_will_consider:
            _require(
                method in {"full_payment", "partial_payment", "installments"},
                f"{profile.user_id}: unknown payment preference {method!r}.",
            )
        _require(profile.current_available_balance >= 0, f"{profile.user_id}: balance cannot be negative.")
        _require(profile.minimum_balance_to_keep >= 0, f"{profile.user_id}: minimum balance cannot be negative.")

    for event in dataset.events:
        _require(event.user_id in profile_user_ids, f"{event.event_id}: normalized event has unknown user.")
        if event.linked_event_id:
            _require(event.linked_event_id in event_ids, f"{event.event_id}: linked_event_id is unknown.")
        if event.amount is not None:
            _require(event.amount >= 0, f"{event.event_id}: amount cannot be negative.")
        if event.minimum_allowed_amount is not None:
            _require(event.minimum_allowed_amount >= 0, f"{event.event_id}: minimum_allowed_amount cannot be negative.")
        if event.status != "unrealized":
            _require(event.settlement_date is not None, f"{event.event_id}: cash event needs settlement_date.")

    for option in dataset.payment_options:
        _require(option.request_id in request_ids, f"{option.payment_option_id}: unknown request_id.")
        _require(option.payment_amount > 0, f"{option.payment_option_id}: payment_amount must be positive.")
        _require(option.number_of_payments > 0, f"{option.payment_option_id}: number_of_payments must be positive.")
        if option.number_of_payments == 1:
            _require(option.payment_frequency_days is None, f"{option.payment_option_id}: single payment has frequency.")
        else:
            _require(
                option.payment_frequency_days is not None and option.payment_frequency_days > 0,
                f"{option.payment_option_id}: installments need positive frequency.",
            )

    for message in dataset.messages:
        _require(message.user_id in profile_user_ids, f"{message.message_id}: unknown user_id.")
        if message.request_id:
            _require(message.request_id in request_ids, f"{message.message_id}: unknown request_id.")
        if message.related_event_id:
            _require(message.related_event_id in event_ids, f"{message.message_id}: unknown related_event_id.")

    for image in dataset.images:
        _require(image.user_id in profile_user_ids, f"{image.image_id}: unknown user_id.")
        if image.request_id:
            _require(image.request_id in request_ids, f"{image.image_id}: unknown request_id.")
        if image.related_event_id:
            _require(image.related_event_id in event_ids, f"{image.image_id}: unknown related_event_id.")
        _require(image.path.exists(), f"{image.image_id}: image path is missing.")


def _decimal(value: str, field_name: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise NormalizationError(f"Invalid decimal for {field_name}: {value!r}") from exc


def _optional_decimal(value: str, field_name: str) -> Decimal | None:
    if not value:
        return None
    return _decimal(value, field_name)


def _int(value: str, field_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise NormalizationError(f"Invalid integer for {field_name}: {value!r}") from exc


def _optional_int(value: str, field_name: str) -> int | None:
    if not value:
        return None
    return _int(value, field_name)


def _date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise NormalizationError(f"Invalid date for {field_name}: {value!r}") from exc


def _optional_date(value: str, field_name: str) -> date | None:
    if not value:
        return None
    return _date(value, field_name)


def _datetime_utc(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NormalizationError(f"Invalid datetime for {field_name}: {value!r}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _currency(value: str, field_name: str) -> str:
    currency = value.upper()
    return _choice(currency, SUPPORTED_CURRENCIES, field_name)


def _choice(value: str, allowed: set[str], field_name: str) -> str:
    if value not in allowed:
        raise NormalizationError(f"Invalid value for {field_name}: {value!r}; expected one of {sorted(allowed)}")
    return value


def _group_by(records, key_fn):
    grouped = {}
    for record in records:
        grouped.setdefault(key_fn(record), []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def _group_nonblank(records, key_fn):
    grouped = {}
    for record in records:
        key = key_fn(record)
        if key:
            grouped.setdefault(key, []).append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise DatasetValidationError(message)

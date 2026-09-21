from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from models import (
    DatasetBundle,
    ExchangeRate,
    FinancialEvent,
    FinancialProfile,
    ImageReference,
    Message,
    OutputTemplateRow,
    PaymentOption,
    Request,
    SampleRequest,
    split_pipe,
)


T = TypeVar("T")


class DatasetValidationError(ValueError):
    pass


SCHEMAS: dict[str, tuple[str, ...]] = {
    "requests.csv": (
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
    ),
    "sample_requests.csv": (
        "request_id",
        "user_id",
        "request_date",
        "request_type",
        "requested_amount",
        "desired_completion_date",
        "allows_partial_payment",
        "request_text",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ),
    "financial_profiles.csv": (
        "user_id",
        "home_currency",
        "current_available_balance",
        "minimum_balance_to_keep",
        "financial_priorities",
        "expense_categories_to_protect",
        "expense_categories_user_is_willing_to_reduce",
        "expense_categories_user_is_willing_to_stop",
        "payment_methods_user_will_consider",
        "max_installment_months",
    ),
    "financial_events.csv": (
        "event_id",
        "user_id",
        "event_type",
        "description",
        "category",
        "direction",
        "amount",
        "currency",
        "event_date",
        "settlement_date",
        "status",
        "linked_event_id",
        "flexibility",
        "minimum_allowed_amount",
    ),
    "exchange_rates.csv": ("rate_date", "from_currency", "to_currency", "rate"),
    "request_payment_options.csv": (
        "payment_option_id",
        "request_id",
        "payment_method",
        "payment_amount",
        "number_of_payments",
        "first_payment_date",
        "payment_frequency_days",
        "financing_fee",
        "total_payable_amount",
    ),
    "messages.csv": (
        "message_id",
        "user_id",
        "request_id",
        "related_event_id",
        "sent_at",
        "source_type",
        "message_text",
    ),
    "images.csv": ("image_id", "user_id", "request_id", "related_event_id"),
    "output.csv": (
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    ),
}


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent

    for path in (current, *current.parents):
        if (path / "AGENTS.md").exists() and (path / "dataset").is_dir():
            return path

    raise DatasetValidationError("Could not locate repo root containing AGENTS.md and dataset/.")


def load_dataset(repo_root: Path | None = None) -> DatasetBundle:
    root = (repo_root or find_repo_root()).resolve()
    dataset_dir = root / "dataset"
    if not dataset_dir.is_dir():
        raise DatasetValidationError(f"Dataset directory does not exist: {dataset_dir}")

    requests = _load_csv(dataset_dir / "requests.csv", SCHEMAS["requests.csv"], _request_from_row)
    sample_requests = _load_csv(
        dataset_dir / "sample_requests.csv",
        SCHEMAS["sample_requests.csv"],
        _sample_request_from_row,
    )
    financial_profiles = _load_csv(
        dataset_dir / "financial_profiles.csv",
        SCHEMAS["financial_profiles.csv"],
        _profile_from_row,
    )
    financial_events = _load_csv(
        dataset_dir / "financial_events.csv",
        SCHEMAS["financial_events.csv"],
        _event_from_row,
    )
    exchange_rates = _load_csv(
        dataset_dir / "exchange_rates.csv",
        SCHEMAS["exchange_rates.csv"],
        lambda row: ExchangeRate(**row),
    )
    payment_options = _load_csv(
        dataset_dir / "request_payment_options.csv",
        SCHEMAS["request_payment_options.csv"],
        lambda row: PaymentOption(**row),
    )
    messages = _load_csv(dataset_dir / "messages.csv", SCHEMAS["messages.csv"], lambda row: Message(**row))
    images = _load_csv(dataset_dir / "images.csv", SCHEMAS["images.csv"], lambda row: ImageReference(**row))
    output_template = _load_csv(
        dataset_dir / "output.csv",
        SCHEMAS["output.csv"],
        lambda row: OutputTemplateRow(**row),
    )

    profiles_by_user_id = _unique_by(financial_profiles, "user_id", "financial_profiles.csv")
    requests_by_request_id = _unique_by(requests, "request_id", "requests.csv")
    events_by_event_id = _unique_by(financial_events, "event_id", "financial_events.csv")
    exchange_rates_by_key = _unique_by_key(
        exchange_rates,
        lambda rate: (rate.rate_date, rate.from_currency, rate.to_currency),
        "exchange_rates.csv",
    )

    bundle = DatasetBundle(
        repo_root=root,
        requests=requests,
        sample_requests=sample_requests,
        financial_profiles=financial_profiles,
        financial_events=financial_events,
        exchange_rates=exchange_rates,
        payment_options=payment_options,
        messages=messages,
        images=images,
        output_template=output_template,
        profiles_by_user_id=profiles_by_user_id,
        requests_by_request_id=requests_by_request_id,
        events_by_event_id=events_by_event_id,
        events_by_user_id=_group_by(financial_events, lambda event: event.user_id),
        payment_options_by_request_id=_group_by(payment_options, lambda option: option.request_id),
        messages_by_user_id=_group_by(messages, lambda message: message.user_id),
        messages_by_request_id=_group_nonblank(messages, lambda message: message.request_id),
        messages_by_event_id=_group_nonblank(messages, lambda message: message.related_event_id),
        images_by_request_id=_group_nonblank(images, lambda image: image.request_id),
        images_by_event_id=_group_nonblank(images, lambda image: image.related_event_id),
        exchange_rates_by_key=exchange_rates_by_key,
    )

    validate_dataset(bundle)
    return bundle


def validate_dataset(bundle: DatasetBundle) -> None:
    _require(bundle.requests, "requests.csv must contain at least one request.")
    _require(bundle.financial_profiles, "financial_profiles.csv must contain at least one profile.")
    _require(bundle.financial_events, "financial_events.csv must contain at least one event.")

    request_ids = {request.request_id for request in bundle.requests}
    sample_request_ids = {request.request_id for request in bundle.sample_requests}
    known_request_ids = request_ids | sample_request_ids
    profile_user_ids = set(bundle.profiles_by_user_id)
    event_ids = set(bundle.events_by_event_id)

    missing_profile_users = sorted(
        ({request.user_id for request in bundle.requests} | {request.user_id for request in bundle.sample_requests})
        - profile_user_ids
    )
    _require(not missing_profile_users, f"Requests reference missing profiles: {missing_profile_users[:10]}")

    template_ids = [row.request_id for row in bundle.output_template]
    _require(
        set(template_ids) == request_ids and len(template_ids) == len(request_ids),
        "dataset/output.csv must contain exactly one row for every request_id in requests.csv.",
    )

    for request in bundle.requests:
        options = bundle.payment_options_by_request_id.get(request.request_id, ())
        _require(options, f"Request {request.request_id} has no payment options.")
        _require(
            2 <= len(options) <= 4,
            f"Request {request.request_id} must have 2 to 4 payment options; found {len(options)}.",
        )

    missing_option_requests = sorted({option.request_id for option in bundle.payment_options} - known_request_ids)
    _require(not missing_option_requests, f"Payment options reference unknown requests: {missing_option_requests[:10]}")

    for event in bundle.financial_events:
        if event.linked_event_id:
            _require(
                event.linked_event_id in event_ids,
                f"Event {event.event_id} links to unknown event {event.linked_event_id}.",
            )
        _require(event.user_id in profile_user_ids, f"Event {event.event_id} references unknown user {event.user_id}.")

    for message in bundle.messages:
        _require(message.user_id in profile_user_ids, f"Message {message.message_id} references unknown user.")
        if message.request_id:
            _require(message.request_id in known_request_ids, f"Message {message.message_id} references unknown request.")
        if message.related_event_id:
            _require(message.related_event_id in event_ids, f"Message {message.message_id} references unknown event.")

    blank_amount_events = {event.event_id for event in bundle.financial_events if not event.amount.strip()}
    image_event_ids = set(bundle.images_by_event_id)
    missing_image_amounts = sorted(blank_amount_events - image_event_ids)
    _require(
        not missing_image_amounts,
        f"Blank-amount events need matching images: {missing_image_amounts[:10]}",
    )

    for image in bundle.images:
        _require(image.user_id in profile_user_ids, f"Image {image.image_id} references unknown user.")
        if image.request_id:
            _require(image.request_id in known_request_ids, f"Image {image.image_id} references unknown request.")
        if image.related_event_id:
            _require(image.related_event_id in event_ids, f"Image {image.image_id} references unknown event.")
        image_path = bundle.repo_root / image.relative_path
        _require(image_path.exists(), f"Image file is missing: {image_path}")


def dataset_summary(bundle: DatasetBundle) -> dict[str, int]:
    blank_amount_events = sum(1 for event in bundle.financial_events if not event.amount.strip())
    return {
        "requests": len(bundle.requests),
        "sample_requests": len(bundle.sample_requests),
        "financial_profiles": len(bundle.financial_profiles),
        "financial_events": len(bundle.financial_events),
        "exchange_rates": len(bundle.exchange_rates),
        "payment_options": len(bundle.payment_options),
        "messages": len(bundle.messages),
        "images": len(bundle.images),
        "output_template_rows": len(bundle.output_template),
        "blank_amount_events": blank_amount_events,
    }


def _load_csv(path: Path, expected_columns: tuple[str, ...], factory: Callable[[dict[str, str]], T]) -> tuple[T, ...]:
    if not path.exists():
        raise DatasetValidationError(f"Required dataset file is missing: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != expected_columns:
            raise DatasetValidationError(
                f"{path.name} columns mismatch. Expected {expected_columns}; found {actual_columns}."
            )
        return tuple(factory({key: value or "" for key, value in row.items()}) for row in reader)


def _request_from_row(row: dict[str, str]) -> Request:
    return Request(
        request_id=row["request_id"],
        user_id=row["user_id"],
        request_date=row["request_date"],
        request_type=row["request_type"],
        requested_amount=row["requested_amount"],
        desired_completion_date=row["desired_completion_date"],
        allows_partial_payment=_parse_bool(row["allows_partial_payment"], "allows_partial_payment"),
        request_text=row["request_text"],
    )


def _sample_request_from_row(row: dict[str, str]) -> SampleRequest:
    return SampleRequest(
        request_id=row["request_id"],
        user_id=row["user_id"],
        request_date=row["request_date"],
        request_type=row["request_type"],
        requested_amount=row["requested_amount"],
        desired_completion_date=row["desired_completion_date"],
        allows_partial_payment=_parse_bool(row["allows_partial_payment"], "allows_partial_payment"),
        request_text=row["request_text"],
        amount_safe_to_pay=row["amount_safe_to_pay"],
        affordability_status=row["affordability_status"],
        recommended_payment_method=row["recommended_payment_method"],
        payment_plan=row["payment_plan"],
        earliest_date_for_full_payment=row["earliest_date_for_full_payment"],
        spending_changes_needed=row["spending_changes_needed"],
        decision_explanation=row["decision_explanation"],
    )


def _profile_from_row(row: dict[str, str]) -> FinancialProfile:
    return FinancialProfile(
        user_id=row["user_id"],
        home_currency=row["home_currency"],
        current_available_balance=row["current_available_balance"],
        minimum_balance_to_keep=row["minimum_balance_to_keep"],
        financial_priorities=split_pipe(row["financial_priorities"]),
        expense_categories_to_protect=split_pipe(row["expense_categories_to_protect"]),
        expense_categories_user_is_willing_to_reduce=split_pipe(row["expense_categories_user_is_willing_to_reduce"]),
        expense_categories_user_is_willing_to_stop=split_pipe(row["expense_categories_user_is_willing_to_stop"]),
        payment_methods_user_will_consider=split_pipe(row["payment_methods_user_will_consider"]),
        max_installment_months=row["max_installment_months"],
    )


def _event_from_row(row: dict[str, str]) -> FinancialEvent:
    return FinancialEvent(**row)


def _parse_bool(value: str, field_name: str) -> bool:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    raise DatasetValidationError(f"Invalid boolean for {field_name}: {value!r}")


def _unique_by(records: Iterable[T], attr: str, source_name: str) -> dict[str, T]:
    return _unique_by_key(records, lambda record: str(getattr(record, attr)), source_name)


def _unique_by_key(records: Iterable[T], key_fn: Callable[[T], str | tuple[str, ...]], source_name: str) -> dict:
    by_key = {}
    for record in records:
        key = key_fn(record)
        if key in by_key:
            raise DatasetValidationError(f"Duplicate key {key!r} in {source_name}.")
        by_key[key] = record
    return by_key


def _group_by(records: Iterable[T], key_fn: Callable[[T], str]) -> dict[str, tuple[T, ...]]:
    grouped: defaultdict[str, list[T]] = defaultdict(list)
    for record in records:
        grouped[key_fn(record)].append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def _group_nonblank(records: Iterable[T], key_fn: Callable[[T], str]) -> dict[str, tuple[T, ...]]:
    grouped: defaultdict[str, list[T]] = defaultdict(list)
    for record in records:
        key = key_fn(record)
        if key:
            grouped[key].append(record)
    return {key: tuple(value) for key, value in grouped.items()}


def _require(condition: object, message: str) -> None:
    if not condition:
        raise DatasetValidationError(message)

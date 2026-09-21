from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

from data_loader import DatasetValidationError, SCHEMAS, load_dataset
from models import DatasetBundle, FinancialEvent, PaymentOption, Request, SampleRequest


OUTPUT_COLUMNS = SCHEMAS["output.csv"]
AFFORDABILITY_STATUSES = {
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
}
PAYMENT_METHODS = {
    "full_payment",
    "partial_payment",
    "installments",
    "wait",
    "not_recommended",
}
PAYMENT_ENTRY_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}):(.+)$")
STOP_RE = re.compile(r"^stop:([^:|]+)$")
REDUCE_RE = re.compile(r"^reduce_to:([^:|]+):(.+)$")


class OutputValidationError(ValueError):
    pass


@dataclass(frozen=True)
class DecisionRow:
    request_id: str
    amount_safe_to_pay: str
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True)
class Payment:
    payment_date: date
    amount: Decimal
    raw_amount: str


def load_decision_rows(path: Path) -> tuple[DecisionRow, ...]:
    if not path.exists():
        raise OutputValidationError(f"Output file does not exist: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        actual_columns = tuple(reader.fieldnames or ())
        if actual_columns != OUTPUT_COLUMNS:
            raise OutputValidationError(f"{path} columns mismatch. Expected {OUTPUT_COLUMNS}; found {actual_columns}.")
        return tuple(DecisionRow(**{key: value or "" for key, value in row.items()}) for row in reader)


def sample_rows_as_decisions(samples: Iterable[SampleRequest]) -> tuple[DecisionRow, ...]:
    return tuple(
        DecisionRow(
            request_id=row.request_id,
            amount_safe_to_pay=row.amount_safe_to_pay,
            affordability_status=row.affordability_status,
            recommended_payment_method=row.recommended_payment_method,
            payment_plan=row.payment_plan,
            earliest_date_for_full_payment=row.earliest_date_for_full_payment,
            spending_changes_needed=row.spending_changes_needed,
            decision_explanation=row.decision_explanation,
        )
        for row in samples
    )


def validate_decisions(
    bundle: DatasetBundle,
    rows: Iterable[DecisionRow],
    include_samples: bool = False,
    expected_request_ids: set[str] | None = None,
) -> None:
    row_list = tuple(rows)
    requests_by_id = _requests_by_id(bundle, samples_only=include_samples)
    expected_ids = expected_request_ids if expected_request_ids is not None else set(requests_by_id)
    unknown_expected_ids = expected_ids - set(requests_by_id)
    _require(not unknown_expected_ids, f"Expected request IDs are unknown: {sorted(unknown_expected_ids)[:10]}")
    row_ids = [row.request_id for row in row_list]

    _require(len(row_ids) == len(set(row_ids)), "Output contains duplicate request_id values.")
    _require(
        set(row_ids) == expected_ids and len(row_ids) == len(expected_ids),
        "Output must contain exactly one row for every expected request_id.",
    )

    for row in row_list:
        request = requests_by_id[row.request_id]
        _validate_decision_row(bundle, row, request)


def validate_output_file(
    path: Path,
    repo_root: Path | None = None,
    expected_request_ids: set[str] | None = None,
) -> None:
    bundle = load_dataset(repo_root)
    rows = load_decision_rows(path)
    validate_decisions(bundle, rows, include_samples=False, expected_request_ids=expected_request_ids)


def _validate_decision_row(bundle: DatasetBundle, row: DecisionRow, request: Request | SampleRequest) -> None:
    requested_amount = _parse_decimal(request.requested_amount, f"{row.request_id}.requested_amount")
    safe_amount = _parse_decimal(row.amount_safe_to_pay, f"{row.request_id}.amount_safe_to_pay")
    request_date = _parse_date(request.request_date, f"{row.request_id}.request_date")
    desired_completion_date = _parse_date(request.desired_completion_date, f"{row.request_id}.desired_completion_date")

    _require(Decimal("0") <= safe_amount <= requested_amount, f"{row.request_id}: amount_safe_to_pay is out of bounds.")
    _require(row.affordability_status in AFFORDABILITY_STATUSES, f"{row.request_id}: invalid affordability_status.")
    _require(row.recommended_payment_method in PAYMENT_METHODS, f"{row.request_id}: invalid recommended_payment_method.")
    _require(row.decision_explanation.strip(), f"{row.request_id}: decision_explanation is required.")

    earliest_date = None
    if row.earliest_date_for_full_payment:
        earliest_date = _parse_date(row.earliest_date_for_full_payment, f"{row.request_id}.earliest_date_for_full_payment")

    payments = _parse_payment_plan(row.payment_plan, row.request_id)
    _validate_status_and_method(row, request_date, earliest_date, payments)
    _validate_full_payment(bundle, row, request, requested_amount, request_date, payments)
    _validate_partial_payment(bundle, row, request, requested_amount, safe_amount, request_date, desired_completion_date, payments)
    _validate_installments(bundle, row, request, payments)
    _validate_wait(bundle, row, request, requested_amount, desired_completion_date, payments, earliest_date)
    _validate_spending_changes(bundle, row)


def _validate_status_and_method(
    row: DecisionRow,
    request_date: date,
    earliest_date: date | None,
    payments: tuple[Payment, ...],
) -> None:
    if row.affordability_status == "affordable_now":
        _require(
            earliest_date == request_date,
            f"{row.request_id}: affordable_now requires earliest_date_for_full_payment to equal request_date.",
        )
        _require(
            row.recommended_payment_method == "full_payment",
            f"{row.request_id}: affordable_now requires full_payment.",
        )

    if row.affordability_status == "not_affordable":
        _require(row.recommended_payment_method == "not_recommended", f"{row.request_id}: not_affordable method mismatch.")
        _require(not payments, f"{row.request_id}: not_affordable requires payment_plan=none.")

    if row.recommended_payment_method == "not_recommended":
        _require(row.payment_plan == "none", f"{row.request_id}: not_recommended requires payment_plan=none.")


def _validate_full_payment(
    bundle: DatasetBundle,
    row: DecisionRow,
    request: Request | SampleRequest,
    requested_amount: Decimal,
    request_date: date,
    payments: tuple[Payment, ...],
) -> None:
    if row.recommended_payment_method != "full_payment":
        return

    profile = bundle.profiles_by_user_id[request.user_id]
    _require("full_payment" in profile.payment_methods_user_will_consider, f"{row.request_id}: user does not accept full_payment.")
    _require(len(payments) == 1, f"{row.request_id}: full_payment requires exactly one payment.")
    _require(payments[0].amount == requested_amount, f"{row.request_id}: full_payment amount must equal requested_amount.")
    _require(payments[0].payment_date >= request_date, f"{row.request_id}: full_payment cannot occur before request_date.")


def _validate_partial_payment(
    bundle: DatasetBundle,
    row: DecisionRow,
    request: Request | SampleRequest,
    requested_amount: Decimal,
    safe_amount: Decimal,
    request_date: date,
    desired_completion_date: date,
    payments: tuple[Payment, ...],
) -> None:
    if row.recommended_payment_method != "partial_payment":
        return

    profile = bundle.profiles_by_user_id[request.user_id]
    _require(row.affordability_status == "affordable_with_plan", f"{row.request_id}: partial_payment status mismatch.")
    _require(
        "partial_payment" in profile.payment_methods_user_will_consider,
        f"{row.request_id}: user does not accept partial_payment.",
    )
    _require(request.allows_partial_payment, f"{row.request_id}: partial_payment is not allowed by request.")
    _require(Decimal("0") < safe_amount < requested_amount, f"{row.request_id}: partial_payment safe amount bounds invalid.")
    _require(len(payments) == 2, f"{row.request_id}: partial_payment must have exactly two payments.")
    _require(payments[0].payment_date == request_date, f"{row.request_id}: first partial payment must be on request_date.")
    _require(payments[0].amount == safe_amount, f"{row.request_id}: first partial payment must equal amount_safe_to_pay.")
    _require(sum((payment.amount for payment in payments), Decimal("0")) == requested_amount, f"{row.request_id}: partial payments must sum to requested_amount.")
    _require(
        payments[1].payment_date <= desired_completion_date,
        f"{row.request_id}: partial_payment completes after desired_completion_date.",
    )


def _validate_installments(
    bundle: DatasetBundle,
    row: DecisionRow,
    request: Request | SampleRequest,
    payments: tuple[Payment, ...],
) -> None:
    if row.recommended_payment_method != "installments":
        return

    matching_options = []
    for option in bundle.payment_options_by_request_id.get(row.request_id, ()):
        if option.payment_method != "installments":
            continue
        if payments == _payments_for_option(option):
            matching_options.append(option)

    _require(matching_options, f"{row.request_id}: installment plan does not match a supplied payment option.")

    profile = bundle.profiles_by_user_id[request.user_id]
    _require(
        "installments" in profile.payment_methods_user_will_consider,
        f"{row.request_id}: user does not accept installments.",
    )
    if profile.max_installment_months:
        max_days = int(profile.max_installment_months) * 31
        _require(
            (payments[-1].payment_date - payments[0].payment_date).days <= max_days,
            f"{row.request_id}: installment plan exceeds max_installment_months.",
        )


def _validate_wait(
    bundle: DatasetBundle,
    row: DecisionRow,
    request: Request | SampleRequest,
    requested_amount: Decimal,
    desired_completion_date: date,
    payments: tuple[Payment, ...],
    earliest_date: date | None,
) -> None:
    if row.recommended_payment_method != "wait":
        return

    _require(row.affordability_status == "affordable_later", f"{row.request_id}: wait status mismatch.")
    _require(earliest_date is not None, f"{row.request_id}: wait requires earliest_date_for_full_payment.")
    _require(len(payments) == 1, f"{row.request_id}: wait requires exactly one payment.")
    _require(payments[0].amount == requested_amount, f"{row.request_id}: wait payment must equal requested_amount.")
    _require(payments[0].payment_date == earliest_date, f"{row.request_id}: wait payment date must equal earliest date.")
    _require(payments[0].payment_date <= desired_completion_date, f"{row.request_id}: wait completes after deadline.")
    profile = bundle.profiles_by_user_id[request.user_id]
    _require("full_payment" in profile.payment_methods_user_will_consider, f"{row.request_id}: wait requires full_payment preference.")


def _validate_spending_changes(bundle: DatasetBundle, row: DecisionRow) -> None:
    if row.spending_changes_needed == "none":
        return

    changes = row.spending_changes_needed.split("|")
    _require(1 <= len(changes) <= 3, f"{row.request_id}: spending_changes_needed must contain 1 to 3 changes.")

    seen_events: set[str] = set()
    for change in changes:
        stop_match = STOP_RE.match(change)
        reduce_match = REDUCE_RE.match(change)
        _require(stop_match or reduce_match, f"{row.request_id}: invalid spending change syntax {change!r}.")

        if stop_match:
            event_id = stop_match.group(1)
            event = _event_for_change(bundle, row.request_id, event_id)
            profile = _profile_for_change(bundle, row.request_id)
            _require(event.category not in profile.expense_categories_to_protect, f"{row.request_id}: cannot stop protected category.")
            _require(
                event.category in profile.expense_categories_user_is_willing_to_stop,
                f"{row.request_id}: user is not willing to stop this category.",
            )
            _require(event.flexibility in {"stoppable", "reducible_or_stoppable"}, f"{row.request_id}: event is not stoppable.")
        else:
            event_id = reduce_match.group(1)
            event = _event_for_change(bundle, row.request_id, event_id)
            profile = _profile_for_change(bundle, row.request_id)
            new_amount = _parse_decimal(reduce_match.group(2), f"{row.request_id}.{event_id}.reduce_to")
            _require(event.category not in profile.expense_categories_to_protect, f"{row.request_id}: cannot reduce protected category.")
            _require(
                event.category in profile.expense_categories_user_is_willing_to_reduce,
                f"{row.request_id}: user is not willing to reduce this category.",
            )
            _require(
                event.flexibility in {"reducible", "reducible_or_stoppable"},
                f"{row.request_id}: event is not reducible.",
            )
            if event.minimum_allowed_amount:
                minimum = _parse_decimal(event.minimum_allowed_amount, f"{row.request_id}.{event_id}.minimum_allowed_amount")
                _require(new_amount >= minimum, f"{row.request_id}: reduce_to is below minimum_allowed_amount.")

        _require(event_id not in seen_events, f"{row.request_id}: cannot change the same event more than once.")
        seen_events.add(event_id)


def _event_for_change(bundle: DatasetBundle, request_id: str, event_id: str) -> FinancialEvent:
    event = bundle.events_by_event_id.get(event_id)
    _require(event is not None, f"{request_id}: spending change references unknown event {event_id}.")

    request = bundle.requests_by_request_id.get(request_id)
    sample_request = next((sample for sample in bundle.sample_requests if sample.request_id == request_id), None)
    user_id = request.user_id if request else sample_request.user_id if sample_request else ""
    _require(event.user_id == user_id, f"{request_id}: spending change references another user's event.")
    _require(event.direction == "debit", f"{request_id}: spending change must target a debit event.")
    return event


def _profile_for_change(bundle: DatasetBundle, request_id: str):
    request = bundle.requests_by_request_id.get(request_id)
    sample_request = next((sample for sample in bundle.sample_requests if sample.request_id == request_id), None)
    user_id = request.user_id if request else sample_request.user_id if sample_request else ""
    profile = bundle.profiles_by_user_id.get(user_id)
    _require(profile is not None, f"{request_id}: cannot resolve profile for spending change.")
    return profile


def _parse_payment_plan(plan: str, request_id: str) -> tuple[Payment, ...]:
    if plan == "none":
        return ()
    _require(plan.strip() == plan and plan, f"{request_id}: invalid payment_plan.")

    payments = []
    previous_date = None
    for entry in plan.split("|"):
        match = PAYMENT_ENTRY_RE.match(entry)
        _require(match, f"{request_id}: invalid payment entry {entry!r}.")
        payment_date = _parse_date(match.group(1), f"{request_id}.payment_plan.date")
        amount = _parse_decimal(match.group(2), f"{request_id}.payment_plan.amount")
        _require(amount > Decimal("0"), f"{request_id}: payment amount must be positive.")
        if previous_date is not None:
            _require(payment_date >= previous_date, f"{request_id}: payment_plan must be chronological.")
        previous_date = payment_date
        payments.append(Payment(payment_date=payment_date, amount=amount, raw_amount=match.group(2)))
    return tuple(payments)


def _payments_for_option(option: PaymentOption) -> tuple[Payment, ...]:
    first_date = _parse_date(option.first_payment_date, f"{option.payment_option_id}.first_payment_date")
    amount = _parse_decimal(option.payment_amount, f"{option.payment_option_id}.payment_amount")
    count = int(option.number_of_payments)
    frequency = int(option.payment_frequency_days or "0")
    return tuple(
        Payment(payment_date=first_date + timedelta(days=frequency * index), amount=amount, raw_amount=option.payment_amount)
        for index in range(count)
    )


def _requests_by_id(bundle: DatasetBundle, samples_only: bool) -> dict[str, Request | SampleRequest]:
    if samples_only:
        return {request.request_id: request for request in bundle.sample_requests}
    return dict(bundle.requests_by_request_id)


def _parse_decimal(value: str, field_name: str) -> Decimal:
    try:
        return Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise OutputValidationError(f"Invalid decimal for {field_name}: {value!r}") from exc


def _parse_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise OutputValidationError(f"Invalid date for {field_name}: {value!r}") from exc


def _require(condition: object, message: str) -> None:
    if not condition:
        raise OutputValidationError(message)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    bundle = load_dataset(repo_root)
    validate_decisions(bundle, sample_rows_as_decisions(bundle.sample_requests), include_samples=True)
    print("Sample output rows validated.")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Sequence
from itertools import combinations, product

from forecast import (
    Forecast,
    ForecastFlow,
    apply_spending_changes,
    build_forecast,
    earliest_full_payment_date,
    max_safe_payment,
    simulate,
)
from money import format_amount, format_amount_for_option
from normalization import NormalizedFinancialProfile, NormalizedPaymentOption, NormalizedRequest
from reconstruction import RecurringSeries, UserFinancialState


@dataclass(frozen=True)
class SpendingChange:
    action: str
    event_id: str
    new_amount: Decimal | None
    description: str = ""
    savings: Decimal = Decimal("0")

    def render(self) -> str:
        if self.action == "stop":
            return f"stop:{self.event_id}"
        return f"reduce_to:{self.event_id}:{format_amount(self.new_amount or Decimal('0'))}"


@dataclass(frozen=True)
class Candidate:
    method: str
    status: str
    payments: tuple[tuple[date, Decimal, str], ...]
    total_paid: Decimal
    start_date: date | None
    completes_by_deadline: bool
    spending_changes: tuple[SpendingChange, ...]
    payment_option_id: str
    lowest_balance: Decimal


def decide_for_request(
    request: NormalizedRequest,
    profile: NormalizedFinancialProfile,
    state: UserFinancialState,
    options: Sequence[NormalizedPaymentOption],
) -> tuple[object, Decimal, date | None, Candidate]:
    forecast = build_forecast(state, request.request_date)
    safe_today = max_safe_payment(forecast, request.requested_amount)
    earliest = earliest_full_payment_date(forecast, request.requested_amount)

    change_sets = _spending_change_sets(profile, state, forecast)
    candidates: list[Candidate] = []
    for changes in change_sets:
        flows = apply_spending_changes(
            forecast.flows,
            [(item.action, item.event_id, item.new_amount) for item in changes],
            request.request_date,
        )
        if changes:
            changed_safe = max_safe_payment(forecast, request.requested_amount, flows)
            changed_earliest = earliest_full_payment_date(forecast, request.requested_amount, flows)
        else:
            changed_safe = safe_today
            changed_earliest = earliest
        candidates.extend(
            _build_candidates(
                request,
                profile,
                options,
                forecast,
                flows,
                changes,
                changed_safe,
                changed_earliest,
            )
        )

    ranked = _rank(candidates)
    if ranked:
        winner = ranked[0]
    else:
        winner = _not_recommended(forecast, request)
    return forecast, safe_today, earliest, winner


def _build_candidates(
    request: NormalizedRequest,
    profile: NormalizedFinancialProfile,
    options: Sequence[NormalizedPaymentOption],
    forecast: Forecast,
    flows: Sequence[ForecastFlow],
    changes: tuple[SpendingChange, ...],
    safe_today: Decimal,
    earliest: date | None,
) -> list[Candidate]:
    prefs = set(profile.payment_methods_user_will_consider)
    deadline = request.desired_completion_date
    amount = request.requested_amount
    out: list[Candidate] = []

    if "full_payment" in prefs:
        ok, low = simulate(forecast, ((request.request_date, amount),), flows, horizon=max(forecast.horizon, request.request_date))
        if ok:
            status = "affordable_now" if not changes else "affordable_with_plan"
            out.append(
                Candidate(
                    method="full_payment",
                    status=status,
                    payments=((request.request_date, amount, format_amount(amount)),),
                    total_paid=amount,
                    start_date=request.request_date,
                    completes_by_deadline=request.request_date <= deadline,
                    spending_changes=changes,
                    payment_option_id=_option_id(options, "full_payment", amount, 1) or "full",
                    lowest_balance=low,
                )
            )
        if not changes and earliest is not None and earliest > request.request_date and earliest <= deadline:
            ok_wait, low_wait = simulate(forecast, ((earliest, amount),), flows, horizon=max(forecast.horizon, earliest))
            if ok_wait:
                out.append(
                    Candidate(
                        method="wait",
                        status="affordable_later",
                        payments=((earliest, amount, format_amount(amount)),),
                        total_paid=amount,
                        start_date=earliest,
                        completes_by_deadline=earliest <= deadline,
                        spending_changes=(),
                        payment_option_id="wait",
                        lowest_balance=low_wait,
                    )
                )

    if (
        "partial_payment" in prefs
        and request.allows_partial_payment
        and Decimal("0") < safe_today < amount
        and earliest is not None
        and earliest <= deadline
        and earliest > request.request_date
    ):
        remainder = amount - safe_today
        payments = (
            (request.request_date, safe_today, format_amount(safe_today)),
            (earliest, remainder, format_amount(remainder)),
        )
        ok, low = simulate(
            forecast,
            ((request.request_date, safe_today), (earliest, remainder)),
            flows,
            horizon=max(forecast.horizon, earliest),
        )
        if ok:
            out.append(
                Candidate(
                    method="partial_payment",
                    status="affordable_with_plan",
                    payments=payments,
                    total_paid=amount,
                    start_date=request.request_date,
                    completes_by_deadline=earliest <= deadline,
                    spending_changes=changes,
                    payment_option_id="partial",
                    lowest_balance=low,
                )
            )

    if "installments" in prefs:
        max_days = None
        if profile.max_installment_months is not None:
            max_days = profile.max_installment_months * 31
        for option in options:
            if option.payment_method != "installments":
                continue
            schedule = _installment_schedule(option)
            if not schedule:
                continue
            span = (schedule[-1][0] - schedule[0][0]).days
            if max_days is not None and span > max_days:
                continue
            pay_pairs = [(day, amt) for day, amt, _raw in schedule]
            horizon = max(forecast.horizon, schedule[-1][0])
            ok, low = simulate(forecast, pay_pairs, flows, horizon=horizon)
            if not ok:
                continue
            last_day = schedule[-1][0]
            out.append(
                Candidate(
                    method="installments",
                    status="affordable_with_plan",
                    payments=schedule,
                    total_paid=option.total_payable_amount,
                    start_date=schedule[0][0],
                    completes_by_deadline=last_day <= deadline,
                    spending_changes=changes,
                    payment_option_id=option.payment_option_id,
                    lowest_balance=low,
                )
            )
    return out


def _installment_schedule(option: NormalizedPaymentOption) -> tuple[tuple[date, Decimal, str], ...]:
    count = option.number_of_payments
    freq = option.payment_frequency_days or 0
    raw = format_amount_for_option(option.payment_amount, str(option.payment_amount))
    # Preserve original option decimal string from dataset via format of Decimal.
    raw = format(option.payment_amount, "f").rstrip("0").rstrip(".") if "." in format(option.payment_amount, "f") else format(option.payment_amount, "f")
    # Better: use canonical Decimal string matching validator (Decimal equality).
    raw = _decimal_string(option.payment_amount)
    return tuple(
        (option.first_payment_date + timedelta(days=freq * index), option.payment_amount, raw)
        for index in range(count)
    )


def _decimal_string(amount: Decimal) -> str:
    text = format(amount, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _option_id(options: Sequence[NormalizedPaymentOption], method: str, amount: Decimal, count: int) -> str | None:
    for option in options:
        if option.payment_method == method and option.number_of_payments == count and option.payment_amount == amount:
            return option.payment_option_id
    return None


def _rank(candidates: Sequence[Candidate]) -> list[Candidate]:
    def key(item: Candidate):
        return (
            0 if item.completes_by_deadline else 1,
            0 if not item.spending_changes else 1,
            item.total_paid,
            item.start_date or date.max,
            len(item.payments) if item.payments else 10**6,
            item.payment_option_id,
        )

    return sorted(candidates, key=key)


def _not_recommended(forecast: Forecast, request: NormalizedRequest) -> Candidate:
    return Candidate(
        method="not_recommended",
        status="not_affordable",
        payments=(),
        total_paid=Decimal("0"),
        start_date=None,
        completes_by_deadline=False,
        spending_changes=(),
        payment_option_id="none",
        lowest_balance=forecast.opening_balance,
    )


def _spending_change_sets(
    profile: NormalizedFinancialProfile,
    state: UserFinancialState,
    forecast: Forecast,
) -> list[tuple[SpendingChange, ...]]:
    """Enumerate every permitted combination of up to three distinct expenses.

    The challenge allows up to three flexible recurring changes. We therefore
    avoid the previous top-six heuristic: a valid low-ranked expense can be
    part of the only safe combination.
    """
    eligible = _eligible_changes(profile, forecast.series, state)
    if not eligible:
        return [()]

    by_event: dict[str, list[SpendingChange]] = {}
    for item in eligible:
        by_event.setdefault(item.event_id, []).append(item)

    event_ids = sorted(by_event)
    groups: list[tuple[SpendingChange, ...]] = [()]
    for size in range(1, min(3, len(event_ids)) + 1):
        for ids in combinations(event_ids, size):
            variants = [by_event[event_id] for event_id in ids]
            for chosen in product(*variants):
                groups.append(tuple(chosen))

    # Stable deterministic ordering: no changes first, then lower number of
    # changes, then larger modeled savings, then event ids/actions.
    def key(group: tuple[SpendingChange, ...]):
        savings = sum(item.savings for item in group)
        signature = tuple((item.event_id, item.action, str(item.new_amount or "")) for item in group)
        return (len(group), -savings, signature)

    return sorted(groups, key=key)


def _eligible_changes(
    profile: NormalizedFinancialProfile,
    series: Sequence[RecurringSeries],
    state: UserFinancialState,
) -> list[SpendingChange]:
    protect = set(profile.expense_categories_to_protect)
    can_stop = set(profile.expense_categories_user_is_willing_to_stop)
    can_reduce = set(profile.expense_categories_user_is_willing_to_reduce)
    changes: list[SpendingChange] = []
    for item in series:
        if item.direction != "debit":
            continue
        if item.category in protect:
            continue
        event_id = item.representative_event_id
        event = state.events_by_id.get(event_id)
        flexibility = item.flexibility
        if item.category in can_stop and flexibility in {"stoppable", "reducible_or_stoppable"}:
            changes.append(SpendingChange("stop", event_id, None, item.description, item.typical_amount_home))
        if item.category in can_reduce and flexibility in {"reducible", "reducible_or_stoppable"}:
            minimum = item.minimum_allowed_amount_home
            if minimum is None and event is not None:
                minimum = event.minimum_allowed_amount_home
            if minimum is not None and minimum < item.typical_amount_home:
                changes.append(
                    SpendingChange(
                        "reduce",
                        event_id,
                        minimum,
                        item.description,
                        item.typical_amount_home - minimum,
                    )
                )
    return changes


def _change_value(change: SpendingChange) -> Decimal:
    return change.savings

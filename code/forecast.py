from __future__ import annotations

import calendar
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from statistics import median
from typing import Iterable, Sequence

from reconstruction import (
    CADENCE_RECURRING,
    CASH_IGNORE,
    CASH_INCLUDE,
    CASH_RESERVE,
    RecurringSeries,
    ReconstructedEvent,
    UserFinancialState,
    detect_series,
)


FORECAST_DAYS = 90
ESSENTIAL_VARIABLE_CATEGORIES = {
    "groceries",
    "utilities",
    "transport",
    "healthcare",
    "insurance",
    "rent",
    "housing",
    "family_support",
    "education",
    "debt_repayment",
}


@dataclass(frozen=True)
class ForecastFlow:
    on: date
    signed_amount: Decimal
    category: str
    description: str
    series_id: str | None
    change_event_id: str | None
    flexibility: str
    minimum_allowed: Decimal | None
    source: str


@dataclass(frozen=True)
class Forecast:
    request_date: date
    horizon: date
    opening_balance: Decimal
    minimum_balance_to_keep: Decimal
    flows: tuple[ForecastFlow, ...]
    series: tuple[RecurringSeries, ...]


def _next_monthly(base: date, target_day: int, step: int) -> date:
    year = base.year + (base.month + step - 1) // 12
    month = (base.month + step - 1) % 12 + 1
    max_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(target_day, max_day))


def build_forecast(state: UserFinancialState, request_date: date) -> Forecast:
    horizon = request_date + timedelta(days=FORECAST_DAYS)
    series_map = {s.series_id: s for s in state.series}

    flows: list[ForecastFlow] = []
    occupied: set[tuple[str, date]] = set()

    for event in state.events:
        if event.cash_role not in {CASH_INCLUDE, CASH_RESERVE} or event.amount_home is None:
            continue
        if event.status not in {"scheduled", "pending"}:
            continue
        cash_date = _event_cash_date(event)
        if event.status == "pending" and event.direction == "debit":
            cash_date = max(cash_date, request_date)
        if not (request_date <= cash_date <= horizon):
            continue
        signed = event.amount_home if event.direction == "credit" else -event.amount_home
        series_id = event.series_id
        change_id = _change_event_id(series_map.get(series_id) if series_id else None, event)
        flows.append(
            ForecastFlow(
                on=cash_date,
                signed_amount=signed,
                category=event.category,
                description=event.description,
                series_id=series_id,
                change_event_id=change_id,
                flexibility=event.flexibility,
                minimum_allowed=event.minimum_allowed_amount_home,
                source=f"known:{event.event_id}",
            )
        )
        occupied.add((_occupation_key(series_id or event.description, event.direction), cash_date))
        occupied.add((_occupation_key(event.category + ":" + event.description.lower(), event.direction), cash_date))
        occupied.add((_occupation_key(event.category, event.direction), cash_date))

    overlay_map = _overlay_index(state)
    for series in series_map.values():
        members = [event for event in state.events if event.event_id in set(series.event_ids)]
        if not members:
            continue

        # Forecast from the latest historical occurrence at or before the
        # request date. Future scheduled/settled rows must not rewrite the
        # historical recurrence anchor for an earlier request.
        historical = [event for event in members if _event_cash_date(event) <= request_date]
        known_future = [event for event in members if _event_cash_date(event) > request_date]
        anchor_pool = historical or [min(members, key=lambda item: (_event_cash_date(item), item.event_id))]
        last = max(anchor_pool, key=lambda item: (_event_cash_date(item), item.event_id))

        # Explicit terminal language in a supplied event means the recurring
        # income/expense stream ends after that event.
        terminal = any(
            token in (event.description or "").lower()
            for event in anchor_pool
            for token in ("final employer payroll", "final payroll", "employment ended", "contract ended", "last payroll")
        )
        if terminal and series.direction == "credit":
            continue

        amount = _series_amount(series, last, overlay_map, request_date, state)
        if amount is None:
            continue
        if overlay_map.get("income_series_ended") and series.category == "salary":
            continue

        is_monthly = 28 <= series.period_days <= 31
        target_day = last.event_date.day
        step = 1
        while True:
            if is_monthly:
                occ = _next_monthly(last.event_date, target_day, step)
            else:
                occ = _event_cash_date(last) + timedelta(days=series.period_days * step)
            step += 1
            if occ > horizon:
                break
            if occ < request_date:
                continue

            nearby = (
                _has_nearby(occupied, series.series_id, series.direction, occ, 3)
                or _has_nearby(occupied, series.category, series.direction, occ, 3)
            )
            if not nearby:
                signed = amount if series.direction == "credit" else -amount
                signed = _apply_percent_overlay(signed, series, occ, overlay_map)
                flows.append(
                    ForecastFlow(
                        on=occ,
                        signed_amount=signed,
                        category=series.category,
                        description=series.description,
                        series_id=series.series_id,
                        change_event_id=series.representative_event_id,
                        flexibility=series.flexibility,
                        minimum_allowed=series.minimum_allowed_amount_home,
                        source=f"series:{series.series_id}",
                    )
                )
                occupied.add((_occupation_key(series.series_id, series.direction), occ))
                occupied.add((_occupation_key(series.category, series.direction), occ))

    recurring_cats = {s.category for s in series_map.values() if s.direction == "debit"}
    historical_annotated = [
        event for event in state.events if _event_cash_date(event) < request_date and event.cash_role != CASH_IGNORE
    ]
    var_flows = [
        flow
        for flow in _variable_flows(historical_annotated, request_date, horizon, overlay_map, occupied, state)
        if flow.category not in recurring_cats
    ]
    flows.extend(var_flows)
    flows.extend(_overlay_one_time_credits(state, request_date, horizon, occupied))

    flows.sort(key=lambda item: (item.on, item.signed_amount, item.source))
    return Forecast(
        request_date=request_date,
        horizon=horizon,
        opening_balance=state.current_available_balance,
        minimum_balance_to_keep=state.minimum_balance_to_keep,
        flows=tuple(flows),
        series=tuple(series_map.values()),
    )


def apply_spending_changes(
    flows: Sequence[ForecastFlow],
    changes: Sequence[tuple[str, str, Decimal | None]],
    request_date: date,
) -> tuple[ForecastFlow, ...]:
    """changes: (action, event_id, new_amount|None) with action in {stop, reduce}."""
    stop_ids = {event_id for action, event_id, _ in changes if action == "stop"}
    reduce_map = {event_id: amount for action, event_id, amount in changes if action == "reduce" and amount is not None}
    updated: list[ForecastFlow] = []
    for flow in flows:
        if flow.on < request_date or flow.signed_amount >= 0:
            updated.append(flow)
            continue
        target = flow.change_event_id
        if target in stop_ids:
            continue
        if target in reduce_map:
            new_abs = reduce_map[target]
            updated.append(
                replace(flow, signed_amount=-abs(new_abs), source=flow.source + ":reduced")
            )
            continue
        updated.append(flow)
    return tuple(updated)


def simulate(
    forecast: Forecast,
    payments: Sequence[tuple[date, Decimal]] = (),
    changed_flows: Sequence[ForecastFlow] | None = None,
    horizon: date | None = None,
) -> tuple[bool, Decimal]:
    end = horizon or forecast.horizon
    if end < forecast.request_date:
        end = forecast.request_date
    flows = changed_flows if changed_flows is not None else forecast.flows
    by_day: dict[date, list[tuple[str, Decimal]]] = defaultdict(list)
    for flow in flows:
        if forecast.request_date <= flow.on <= end:
            by_day[flow.on].append(("flow", flow.signed_amount))
    for pay_date, amount in payments:
        if amount > 0 and forecast.request_date <= pay_date <= end:
            by_day[pay_date].append(("pay", -amount))

    balance = forecast.opening_balance
    lowest = balance
    if lowest < forecast.minimum_balance_to_keep:
        return False, lowest

    current = forecast.request_date
    while current <= end:
        items = by_day.get(current, [])
        credits = [value for kind, value in items if value > 0]
        debits = [value for kind, value in items if value < 0]
        for value in credits:
            balance += value
            if balance < lowest:
                lowest = balance
        for value in debits:
            balance += value
            if balance < lowest:
                lowest = balance
            if balance < forecast.minimum_balance_to_keep:
                return False, lowest
        current += timedelta(days=1)
    return True, lowest


def max_safe_payment(
    forecast: Forecast,
    cap: Decimal,
    changed_flows: Sequence[ForecastFlow] | None = None,
) -> Decimal:
    cap = max(Decimal("0"), cap)
    if cap == 0:
        return cap
    safe, _ = simulate(forecast, ((forecast.request_date, cap),), changed_flows)
    if safe:
        return cap
    low = Decimal("0")
    high = cap
    step = Decimal("0.01")
    for _ in range(48):
        mid = ((low + high) / 2).quantize(step)
        ok, _ = simulate(forecast, ((forecast.request_date, mid),), changed_flows)
        if ok:
            low = mid
        else:
            high = mid
        if high - low <= step:
            break
    # Snap down to a safe value.
    while low > 0:
        ok, _ = simulate(forecast, ((forecast.request_date, low),), changed_flows)
        if ok:
            return low
        low -= step
    return Decimal("0")


def earliest_full_payment_date(
    forecast: Forecast,
    amount: Decimal,
    changed_flows: Sequence[ForecastFlow] | None = None,
) -> date | None:
    if amount <= 0:
        return forecast.request_date
    current = forecast.request_date
    while current <= forecast.horizon:
        ok, _ = simulate(forecast, ((current, amount),), changed_flows, horizon=max(forecast.horizon, current))
        if ok:
            return current
        current += timedelta(days=1)
    return None


def _event_cash_date(event: ReconstructedEvent) -> date:
    return event.settlement_date or event.event_date


def _occupation_key(series_or_desc: str, direction: str) -> str:
    return f"{direction}:{series_or_desc}"


def _has_nearby(occupied: set[tuple[str, date]], series_id: str | None, direction: str, occ: date, window: int) -> bool:
    keys = [_occupation_key(series_id or "", direction)]
    for key in keys:
        for delta in range(-window, window + 1):
            if (key, occ + timedelta(days=delta)) in occupied:
                return True
    return False


def _change_event_id(series: RecurringSeries | None, event: ReconstructedEvent) -> str | None:
    if series is not None:
        return series.representative_event_id
    if event.cadence == CADENCE_RECURRING:
        return event.event_id
    return None


def _overlay_index(state: UserFinancialState) -> dict[str, object]:
    by_type: dict[str, object] = {}
    for overlay in state.overlays:
        by_type.setdefault(overlay.overlay_type, overlay)
    return by_type


def _overlay_amount_home(
    overlay,
    series: RecurringSeries,
    last: ReconstructedEvent,
    request_date: date,
    state: UserFinancialState,
) -> Decimal:
    amount = overlay.amount
    currency = getattr(overlay, "currency", None)
    if not currency or currency == state.home_currency:
        return amount
    effective = overlay.effective_date or request_date
    key = (effective, currency, state.home_currency)
    rate = state.exchange_rates_by_key.get(key) if state.exchange_rates_by_key else None
    if rate is not None:
        return amount * rate.rate
    return amount


def _series_amount(series: RecurringSeries, last: ReconstructedEvent, overlays: dict, request_date: date, state: UserFinancialState) -> Decimal | None:
    amount = last.amount_home if last.amount_home is not None else series.typical_amount_home
    salary_increase = overlays.get("salary_increase")
    if series.category == "salary" and salary_increase is not None and getattr(salary_increase, "amount", None) is not None:
        effective = salary_increase.effective_date or request_date
        if effective <= request_date + timedelta(days=FORECAST_DAYS):
            amount = _overlay_amount_home(salary_increase, series, last, request_date, state)
    reduced = overlays.get("temporary_salary_reduction") or overlays.get("household_income_reduced")
    if series.category == "salary" and reduced is not None and getattr(reduced, "amount", None) is not None:
        amount = reduced.amount
    return amount


def _apply_percent_overlay(signed: Decimal, series: RecurringSeries, occ: date, overlays: dict) -> Decimal:
    rent = overlays.get("rent_increase_percent")
    if rent is None or series.category != "rent":
        return signed
    percent = getattr(rent, "percent", None)
    effective = getattr(rent, "effective_date", None)
    if percent is None or effective is None or occ < effective:
        return signed
    factor = Decimal("1") + (percent / Decimal("100"))
    return (signed * factor).quantize(Decimal("0.01"))


def _variable_flows(
    historical: Sequence[ReconstructedEvent],
    request_date: date,
    horizon: date,
    overlays: dict,
    occupied: set[tuple[str, date]],
    state: UserFinancialState,
) -> list[ForecastFlow]:
    lookback_start = request_date - timedelta(days=FORECAST_DAYS)
    buckets: dict[str, list[Decimal]] = defaultdict(list)
    for event in historical:
        if event.direction != "debit" or event.amount_home is None:
            continue
        if event.cadence == CADENCE_RECURRING:
            continue
        cash_date = _event_cash_date(event)
        if not (lookback_start <= cash_date < request_date):
            continue
        if event.category in ESSENTIAL_VARIABLE_CATEGORIES:
            buckets[event.category].append(event.amount_home)

    flows: list[ForecastFlow] = []
    lookback_days = FORECAST_DAYS
    cursor = request_date
    # Weekly conservative debit: 7 * (lookback_sum / lookback_days) per category.
    while cursor <= horizon:
        week_end = min(cursor + timedelta(days=6), horizon)
        for category, amounts in buckets.items():
            if not amounts:
                continue
            daily = sum(amounts, Decimal("0")) / Decimal(lookback_days)
            weekly = (daily * Decimal(week_end.toordinal() - cursor.toordinal() + 1)).quantize(Decimal("0.01"))
            if weekly <= 0:
                continue
            flows.append(
                ForecastFlow(
                    on=cursor,
                    signed_amount=-weekly,
                    category=category,
                    description=f"conservative {category} spend",
                    series_id=None,
                    change_event_id=None,
                    flexibility="fixed",
                    minimum_allowed=None,
                    source=f"variable:{category}",
                )
            )
        cursor += timedelta(days=7)
    return flows


def _overlay_one_time_credits(
    state: UserFinancialState,
    request_date: date,
    horizon: date,
    occupied: set[tuple[str, date]],
) -> list[ForecastFlow]:
    flows: list[ForecastFlow] = []
    for overlay in state.overlays:
        if overlay.overlay_type not in {"confirmed_first_salary", "confirmed_invoice_credit", "one_time_arrears"}:
            continue
        if overlay.amount is None or overlay.effective_date is None:
            continue
        if not (request_date <= overlay.effective_date <= horizon):
            continue
        flows.append(
            ForecastFlow(
                on=overlay.effective_date,
                signed_amount=overlay.amount,
                category="salary" if "salary" in overlay.overlay_type else "income",
                description=overlay.overlay_type,
                series_id=None,
                change_event_id=None,
                flexibility="fixed",
                minimum_allowed=None,
                source=f"overlay:{overlay.message_id}",
            )
        )
    date_shift = next((item for item in state.overlays if item.overlay_type == "salary_date_amendment"), None)
    if date_shift is not None and date_shift.effective_date is not None:
        new_date = date_shift.effective_date
        if request_date <= new_date <= horizon:
            salary_series = next(
                (series for series in state.series if series.category == "salary" and series.direction == "credit"),
                None,
            )
            if salary_series is not None:
                members = [
                    event for event in state.events
                    if event.event_id in set(salary_series.event_ids)
                    and _event_cash_date(event) <= request_date
                ]
                if members:
                    last = max(members, key=lambda event: (_event_cash_date(event), event.event_id))
                    amount = _series_amount(salary_series, last, _overlay_index(state), request_date, state)
                    # Remove one forecast salary occurrence in the amended payroll
                    # window, then place it on the confirmed new date.
                    remaining = [
                        flow for flow in flows
                        if not (
                            flow.category == "salary"
                            and flow.on != new_date
                            and abs((flow.on - new_date).days) <= 20
                        )
                    ]
                    flows = remaining
                    if not any(flow.category == "salary" and flow.on == new_date for flow in flows):
                        flows.append(
                            ForecastFlow(
                                on=new_date,
                                signed_amount=amount,
                                category="salary",
                                description="salary_date_amendment",
                                series_id=salary_series.series_id,
                                change_event_id=None,
                                flexibility="fixed",
                                minimum_allowed=None,
                                source=f"overlay:{date_shift.message_id}",
                            )
                        )
    return flows

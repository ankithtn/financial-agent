from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import date
from decimal import Decimal
from statistics import median
from typing import Iterable, Sequence

from currency import convert_to_home
from evidence import (
    EvidenceOverlay,
    image_amount_for,
    parse_message_overlays,
)
from normalization import (
    NormalizedDataset,
    NormalizedFinancialEvent,
    NormalizedFinancialProfile,
    NormalizedImageReference,
    NormalizedMessage,
)


CASH_INCLUDE = "include"
CASH_RESERVE = "reserve"
CASH_IGNORE = "ignore"

CADENCE_RECURRING = "recurring"
CADENCE_VARIABLE = "variable"
CADENCE_ONE_TIME = "one_time"

VARIABLE_CATEGORIES = {"groceries", "transport", "dining", "utilities", "shopping", "entertainment", "healthcare"}
WINDFALL_CATEGORIES = {"windfall"}


@dataclass(frozen=True)
class ReconstructedEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    original_amount: Decimal | None
    original_currency: str
    amount_home: Decimal | None
    home_currency: str
    event_date: date
    settlement_date: date | None
    status: str
    linked_event_id: str | None
    flexibility: str
    minimum_allowed_amount_home: Decimal | None
    cash_role: str
    ignore_reason: str | None
    cadence: str
    series_id: str | None
    recurrence_days: int | None
    amount_source: str
    provenance: tuple[str, ...]


@dataclass(frozen=True)
class RecurringSeries:
    series_id: str
    user_id: str
    event_type: str
    category: str
    description: str
    direction: str
    period_days: int
    typical_amount_home: Decimal
    last_event_date: date
    flexibility: str
    minimum_allowed_amount_home: Decimal | None
    occurrence_count: int
    event_ids: tuple[str, ...]
    representative_event_id: str


@dataclass(frozen=True)
class UserFinancialState:
    user_id: str
    home_currency: str
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    events: tuple[ReconstructedEvent, ...]
    series: tuple[RecurringSeries, ...]
    overlays: tuple[EvidenceOverlay, ...]
    events_by_id: dict[str, ReconstructedEvent]
    exchange_rates_by_key: dict = field(default_factory=dict)


@dataclass(frozen=True)
class ReconstructedDataset:
    users: dict[str, UserFinancialState]
    events_by_id: dict[str, ReconstructedEvent]


def detect_series(
    events: Sequence[ReconstructedEvent],
) -> tuple[dict[str, RecurringSeries], dict[str, tuple[str | None, str, int | None]]]:
    return _detect_series(events)


def reconstruct_dataset(dataset: NormalizedDataset) -> ReconstructedDataset:
    overlays = parse_message_overlays(dataset.messages)
    overlays_by_user: dict[str, list[EvidenceOverlay]] = defaultdict(list)
    for overlay in overlays:
        overlays_by_user[overlay.user_id].append(overlay)

    users: dict[str, UserFinancialState] = {}
    events_by_id: dict[str, ReconstructedEvent] = {}
    for profile in dataset.profiles:
        state = reconstruct_user_state(
            profile,
            dataset.events_by_user_id.get(profile.user_id, ()),
            exchange_rates_by_key=dataset.exchange_rates_by_key,
            messages=dataset.messages_by_user_id.get(profile.user_id, ()),
            images_by_event_id=dataset.images_by_event_id,
            overlays=tuple(overlays_by_user.get(profile.user_id, ())),
        )
        users[profile.user_id] = state
        events_by_id.update(state.events_by_id)
    return ReconstructedDataset(users=users, events_by_id=events_by_id)


def reconstruct_user_state(
    profile: NormalizedFinancialProfile,
    events: Sequence[NormalizedFinancialEvent],
    *,
    exchange_rates_by_key: dict,
    messages: Sequence[NormalizedMessage] = (),
    images_by_event_id: dict[str, tuple[NormalizedImageReference, ...]] | None = None,
    overlays: Sequence[EvidenceOverlay] | None = None,
) -> UserFinancialState:
    image_index = images_by_event_id or {}
    user_overlays = tuple(overlays if overlays is not None else parse_message_overlays(messages))
    children_by_parent: dict[str, list[NormalizedFinancialEvent]] = defaultdict(list)
    events_by_id = {event.event_id: event for event in events}
    for event in events:
        if event.linked_event_id:
            children_by_parent[event.linked_event_id].append(event)

    reconstructed: list[ReconstructedEvent] = []
    for event in events:
        reconstructed.append(
            _reconstruct_event(
                profile,
                event,
                children_by_parent=children_by_parent,
                events_by_id=events_by_id,
                exchange_rates_by_key=exchange_rates_by_key,
                images_by_event_id=image_index,
                overlays=user_overlays,
            )
        )

    series_map, cadence_by_id = _detect_series(reconstructed)
    annotated = []
    for event in reconstructed:
        series_id, cadence, period = cadence_by_id.get(event.event_id, (None, CADENCE_ONE_TIME, None))
        annotated.append(
            replace(event, cadence=cadence, series_id=series_id, recurrence_days=period)
        )

    by_id = {event.event_id: event for event in annotated}
    return UserFinancialState(
        user_id=profile.user_id,
        home_currency=profile.home_currency,
        current_available_balance=profile.current_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        events=tuple(annotated),
        series=tuple(series_map.values()),
        overlays=user_overlays,
        events_by_id=by_id,
        exchange_rates_by_key=exchange_rates_by_key,
    )


def reconstruction_summary(dataset: ReconstructedDataset) -> dict[str, int]:
    events = tuple(dataset.events_by_id.values())
    return {
        "reconstructed_users": len(dataset.users),
        "reconstructed_events": len(events),
        "cash_include": sum(1 for event in events if event.cash_role == CASH_INCLUDE),
        "cash_reserve": sum(1 for event in events if event.cash_role == CASH_RESERVE),
        "cash_ignore": sum(1 for event in events if event.cash_role == CASH_IGNORE),
        "recurring_events": sum(1 for event in events if event.cadence == CADENCE_RECURRING),
        "variable_events": sum(1 for event in events if event.cadence == CADENCE_VARIABLE),
        "one_time_events": sum(1 for event in events if event.cadence == CADENCE_ONE_TIME),
        "recurring_series": sum(len(user.series) for user in dataset.users.values()),
        "evidence_overlays": sum(len(user.overlays) for user in dataset.users.values()),
        "amounts_from_images": sum(1 for event in events if event.amount_source == "image"),
        "unresolved_amounts": sum(
            1 for event in events if event.amount_home is None and event.cash_role != CASH_IGNORE
        ),
    }


def _reconstruct_event(
    profile: NormalizedFinancialProfile,
    event: NormalizedFinancialEvent,
    *,
    children_by_parent: dict[str, list[NormalizedFinancialEvent]],
    events_by_id: dict[str, NormalizedFinancialEvent],
    exchange_rates_by_key: dict,
    images_by_event_id: dict[str, tuple[NormalizedImageReference, ...]],
    overlays: Sequence[EvidenceOverlay],
) -> ReconstructedEvent:
    amount, amount_source, provenance = _resolve_amount(event, images_by_event_id)
    rate_date = event.settlement_date or event.event_date
    amount_home = None
    if amount is not None:
        amount_home = convert_to_home(
            amount,
            event.currency,
            profile.home_currency,
            rate_date,
            exchange_rates_by_key,
            field_name=f"{event.event_id}.amount",
        )
        if event.currency != profile.home_currency:
            provenance = provenance + (f"fx:{event.currency}->{profile.home_currency}@{rate_date.isoformat()}",)

    minimum_home = None
    if event.minimum_allowed_amount is not None:
        minimum_home = convert_to_home(
            event.minimum_allowed_amount,
            event.currency,
            profile.home_currency,
            rate_date,
            exchange_rates_by_key,
            field_name=f"{event.event_id}.minimum_allowed_amount",
        )

    cash_role, ignore_reason, extra_prov = _cash_role(event, children_by_parent, events_by_id, overlays)
    return ReconstructedEvent(
        event_id=event.event_id,
        user_id=event.user_id,
        event_type=event.event_type,
        description=event.description,
        category=event.category,
        direction=event.direction,
        original_amount=amount,
        original_currency=event.currency,
        amount_home=amount_home,
        home_currency=profile.home_currency,
        event_date=event.event_date,
        settlement_date=event.settlement_date,
        status=event.status,
        linked_event_id=event.linked_event_id,
        flexibility=event.flexibility,
        minimum_allowed_amount_home=minimum_home,
        cash_role=cash_role,
        ignore_reason=ignore_reason,
        cadence=CADENCE_ONE_TIME,
        series_id=None,
        recurrence_days=None,
        amount_source=amount_source,
        provenance=provenance + extra_prov,
    )


def _resolve_amount(
    event: NormalizedFinancialEvent,
    images_by_event_id: dict[str, tuple[NormalizedImageReference, ...]],
) -> tuple[Decimal | None, str, tuple[str, ...]]:
    if event.amount is not None:
        return event.amount, "event", ("amount:event",)
    images = images_by_event_id.get(event.event_id, ())
    for image in images:
        extracted = image_amount_for(image.image_id, image_path=image.path, request_id=image.request_id)
        if extracted is not None:
            return extracted, "image", (f"amount:image:{image.image_id}",)
    return None, "unresolved", ("amount:unresolved",)


def _cash_role(
    event: NormalizedFinancialEvent,
    children_by_parent: dict[str, list[NormalizedFinancialEvent]],
    events_by_id: dict[str, NormalizedFinancialEvent],
    overlays: Sequence[EvidenceOverlay],
) -> tuple[str, str | None, tuple[str, ...]]:
    provenance: list[str] = []
    children = children_by_parent.get(event.event_id, [])

    if event.status == "cancelled":
        return CASH_IGNORE, "cancelled", ("precedence:explicit_cancellation",)
    if event.status == "failed":
        if any(child.status == "scheduled" for child in children):
            provenance.append("precedence:failed_replaced_by_scheduled_retry")
        return CASH_IGNORE, "failed", tuple(provenance or ("precedence:failed",))
    if event.status == "unrealized" or event.direction == "non_cash":
        return CASH_IGNORE, "unrealized_non_cash", ("cash:ignore_unrealized",)
    if event.event_type == "investment_valuation":
        return CASH_IGNORE, "unrealized_non_cash", ("cash:ignore_valuation",)

    parent = events_by_id.get(event.linked_event_id) if event.linked_event_id else None
    if parent and parent.status == "cancelled" and event.status == "settled":
        provenance.append("precedence:settled_amendment_after_cancellation")

    related_overlays = [item for item in overlays if item.related_event_id == event.event_id]
    overlay_types = {item.overlay_type for item in related_overlays} | {
        item.overlay_type for item in overlays if item.related_event_id is None
    }

    if event.status == "pending" and event.direction == "credit":
        return CASH_IGNORE, "pending_credit", ("cash:ignore_pending_credit",)
    if event.status == "pending" and event.direction == "debit":
        reason = "pending_debit"
        if "possible duplicate" in event.description.lower():
            reason = "pending_duplicate_debit"
            provenance.append("safer:reserve_unresolved_duplicate_debit")
        return CASH_RESERVE, reason, tuple(provenance + ["cash:reserve_pending_debit"])

    if event.category in WINDFALL_CATEGORIES and event.status != "settled":
        return CASH_IGNORE, "unconfirmed_windfall", ("cash:ignore_unconfirmed_windfall",)

    if event.direction == "credit" and event.status != "settled":
        if event.status == "scheduled" and event.event_type == "income" and event.category == "salary":
            return CASH_INCLUDE, None, tuple(provenance + ["cash:confirmed_salary_on_settlement_date"])
        if event.status == "scheduled" and event.event_type in {"income", "refund"}:
            # Confirmed scheduled salary already handled. Other scheduled credits stay
            # included only when they are salary; refunds/invoices wait until settled
            # unless an overlay confirms an invoice.
            if event.event_type == "refund":
                return CASH_IGNORE, "unsettled_refund", ("cash:ignore_unsettled_refund",)
        if "pending_credit_unconfirmed" in overlay_types and event.event_type == "income":
            if event.status != "scheduled":
                return CASH_IGNORE, "unconfirmed_income", ("evidence:pending_payout",)

    if event.event_type in {"investment_sale", "refund"} and event.status != "settled":
        return CASH_IGNORE, "unsettled_credit", ("cash:ignore_unsettled_credit",)

    if event.status in {"settled", "scheduled"}:
        return CASH_INCLUDE, None, tuple(provenance + [f"cash:include_{event.status}"])

    return CASH_IGNORE, f"unsupported_status:{event.status}", ("cash:safer_ignore",)


def _detect_series(
    events: Sequence[ReconstructedEvent],
) -> tuple[dict[str, RecurringSeries], dict[str, tuple[str | None, str, int | None]]]:
    groups: dict[tuple[str, str, str, str, str], list[ReconstructedEvent]] = defaultdict(list)
    salary_events: list[ReconstructedEvent] = []
    for event in events:
        if event.cash_role == CASH_IGNORE and event.status in {"cancelled", "failed", "unrealized"}:
            continue
        if event.category == "salary" and event.event_type == "income":
            salary_events.append(event)
            continue
        if (
            event.category in {
                "utilities",
                "rent",
                "insurance",
                "debt_repayment",
                "education",
                "housing",
                "family_support",
            }
            and event.direction == "debit"
        ):
            key = (event.user_id, event.event_type, event.category, event.category, event.direction)
            groups[key].append(event)
            continue
        key = (event.user_id, event.event_type, event.category, event.description.lower(), event.direction)
        groups[key].append(event)

    # Salary streams need finer grouping than category alone. Base salary/payroll
    # and commissions often share category="salary" but have different cadences.
    # Group base-pay records together while keeping commission/bonus streams out
    # of the deterministic recurring-income projection unless they independently
    # prove a stable cadence.
    for event in salary_events:
        desc = event.description.lower()
        if any(token in desc for token in ("commission", "bonus", "arrears", "one-time")):
            key_desc = event.description.lower()
        else:
            key_desc = "base_salary"
        groups[
            (
                event.user_id,
                "income",
                "salary",
                key_desc,
                "credit",
            )
        ].append(event)

    cadence_by_id: dict[str, tuple[str | None, str, int | None]] = {}
    series_map: dict[str, RecurringSeries] = {}

    for key, members in groups.items():
        unique_members = _unique_by_id(members)
        unique_members.sort(key=lambda item: (item.event_date, item.event_id))
        period = _recurrence_period(unique_members)
        cadence = _cadence_for(unique_members, period)
        series_id = None
        if cadence == CADENCE_RECURRING and period is not None:
            series_id = "series:" + "|".join(key)
            amounts = [item.amount_home for item in unique_members if item.amount_home is not None]
            typical = _median_decimal(amounts) if amounts else Decimal("0")
            latest = unique_members[-1]
            series_map[series_id] = RecurringSeries(
                series_id=series_id,
                user_id=latest.user_id,
                event_type=latest.event_type,
                category=latest.category,
                description=latest.description,
                direction=latest.direction,
                period_days=period,
                typical_amount_home=typical,
                last_event_date=latest.event_date,
                flexibility=latest.flexibility,
                minimum_allowed_amount_home=latest.minimum_allowed_amount_home,
                occurrence_count=len(unique_members),
                event_ids=tuple(item.event_id for item in unique_members),
                representative_event_id=latest.event_id,
            )
        for item in unique_members:
            cadence_by_id[item.event_id] = (series_id, cadence, period)
    return series_map, cadence_by_id


def _unique_by_id(events: Sequence[ReconstructedEvent]) -> list[ReconstructedEvent]:
    by_id = {}
    for event in events:
        by_id[event.event_id] = event
    return list(by_id.values())


def _cadence_for(members: Sequence[ReconstructedEvent], period: int | None) -> str:
    if not members:
        return CADENCE_ONE_TIME
    sample = members[0]
    if sample.category in WINDFALL_CATEGORIES or sample.event_type in {"refund", "investment_sale", "investment_purchase"}:
        return CADENCE_ONE_TIME
    if sample.event_type == "subscription" and len(members) >= 2 and period:
        return CADENCE_RECURRING
    if period and len(members) >= 3:
        return CADENCE_RECURRING
    if sample.event_type == "income" and sample.category == "salary" and period and len(members) >= 2:
        return CADENCE_RECURRING
    if sample.category in VARIABLE_CATEGORIES and sample.direction == "debit" and len(members) >= 3:
        return CADENCE_VARIABLE
    if sample.category in VARIABLE_CATEGORIES and sample.direction == "debit":
        # Same description may still be irregular essential spend.
        user_category_count = len(members)
        if user_category_count >= 3 and period is None:
            return CADENCE_VARIABLE
    return CADENCE_ONE_TIME


def _recurrence_period(members: Sequence[ReconstructedEvent]) -> int | None:
    if len(members) < 2:
        return None
    dates = sorted({item.event_date for item in members})
    if len(dates) < 2:
        return None
    diffs = [(dates[index] - dates[index - 1]).days for index in range(1, len(dates))]
    diffs = [value for value in diffs if value > 0]
    if not diffs:
        return None
    mid = median(diffs)
    buckets = (
        (7, 2),
        (10, 2),
        (14, 3),
        (15, 3),
        (21, 3),
        (28, 3),
        (30, 4),
        (31, 4),
        (60, 6),
        (90, 8),
        (365, 15),
    )
    for period, tolerance in buckets:
        if abs(mid - period) <= tolerance:
            # Require most gaps to agree so irregular grocery runs are not monthly bills.
            agreed = sum(1 for value in diffs if abs(value - period) <= tolerance)
            if agreed / len(diffs) >= 0.6:
                return 30 if period in {28, 31} else period
    return None


def _median_decimal(values: Iterable[Decimal]) -> Decimal:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal("2")

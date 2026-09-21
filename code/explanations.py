from __future__ import annotations

from datetime import date
from decimal import Decimal

from money import format_grouped
from normalization import NormalizedRequest
from planner import Candidate
from reconstruction import UserFinancialState


_LLM = None


def configure_explainer_llm(llm) -> None:
    global _LLM
    _LLM = llm


MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def _deterministic_explanation(
    request: NormalizedRequest,
    state: UserFinancialState,
    winner: Candidate,
    safe_today: Decimal,
    earliest: date | None,
) -> str:
    currency = state.home_currency
    minimum = format_grouped(state.minimum_balance_to_keep, currency)
    requested = format_grouped(request.requested_amount, currency)
    deadline = _pretty_date(request.desired_completion_date)
    change_clause = _change_clause(winner, state)

    if winner.method == "not_recommended":
        if safe_today > 0:
            return (
                f"Do not proceed with the {requested} request. Although "
                f"{format_grouped(safe_today, currency)} is available today, the full amount "
                f"cannot be completed safely within 90 days."
            )
        return (
            f"Do not make this payment by {deadline}. None of the available options keeps "
            f"the {minimum} minimum protected."
        )

    if winner.method == "full_payment":
        lead = f"{change_clause}pay {requested} today." if change_clause else f"Pay {requested} today."
        return f"{lead} This leaves at least {minimum} available over the next 90 days."

    if winner.method == "wait" and winner.payments:
        pay_day = winner.payments[0][0]
        return (
            f"Pay {requested} in full on {_pretty_date(pay_day)}. Paying earlier would take "
            f"the balance below the {minimum} minimum."
        )

    if winner.method == "partial_payment" and len(winner.payments) == 2:
        first = format_grouped(winner.payments[0][1], currency)
        second = format_grouped(winner.payments[1][1], currency)
        return (
            f"Pay {first} today and the remaining {second} on {_pretty_date(winner.payments[1][0])}. "
            f"This completes the full request and keeps the {minimum} minimum protected."
        )

    if winner.method == "installments" and winner.payments:
        each = format_grouped(winner.payments[0][1], currency)
        start = _pretty_date(winner.payments[0][0])
        count = len(winner.payments)
        lead = f"{change_clause}use {count} installments of {each}, starting {start}."
        lead = lead[0].upper() + lead[1:]
        return f"{lead} This leaves at least {minimum} available."

    return f"Recommend {winner.method} for {requested} while keeping {minimum} protected."


def explain_decision(
    request: NormalizedRequest,
    state: UserFinancialState,
    winner: Candidate,
    safe_today: Decimal,
    earliest: date | None,
) -> str:
    deterministic = _deterministic_explanation(request, state, winner, safe_today, earliest)
    if _LLM is None or not getattr(_LLM, "enabled", False):
        return deterministic

    payment_plan = "none"
    if winner.payments:
        payment_plan = "|".join(
            f"{day.isoformat()}:{format_grouped(amount, state.home_currency)}" for day, amount, _raw in winner.payments
        )
    changes = "none"
    if winner.spending_changes:
        changes = "|".join(item.render() for item in winner.spending_changes)

    context = (
        f"request_id={request.request_id}\n"
        f"request_text={request.request_text}\n"
        f"home_currency={state.home_currency}\n"
        f"requested_amount={format_grouped(request.requested_amount, state.home_currency)}\n"
        f"amount_safe_today={format_grouped(safe_today, state.home_currency)}\n"
        f"status={winner.status}\n"
        f"method={winner.method}\n"
        f"payment_plan={payment_plan}\n"
        f"earliest_full_payment={earliest.isoformat() if earliest else ''}\n"
        f"spending_changes={changes}\n"
        f"minimum_balance={format_grouped(state.minimum_balance_to_keep, state.home_currency)}\n"
        f"deterministic_explanation={deterministic}\n"
    )
    llm_text = _LLM.explain_decision(request_id=request.request_id, context=context)
    # If the model fails or returns an unusable answer, preserve deterministic output.
    if llm_text and len(llm_text) <= 700:
        return llm_text
    return deterministic


def _change_clause(winner: Candidate, state: UserFinancialState) -> str:
    if not winner.spending_changes:
        return ""
    parts = []
    for change in winner.spending_changes:
        event = state.events_by_id.get(change.event_id)
        name = change.description or (event.description if event else change.event_id)
        name = name[:1].lower() + name[1:] if name else change.event_id
        if change.action == "stop":
            parts.append(f"stop the {name}")
        else:
            amount = format_grouped(change.new_amount or Decimal("0"), state.home_currency)
            parts.append(f"reduce the {name} to {amount}")
    joined = " and ".join(parts) if len(parts) == 2 else ", ".join(parts)
    if len(parts) > 2:
        joined = ", ".join(parts[:-1]) + ", and " + parts[-1]
    return f"{joined[0].upper() + joined[1:]}, then "


def _pretty_date(value: date) -> str:
    return f"{value.day} {MONTHS[value.month - 1]} {value.year}"

from __future__ import annotations

from typing import Sequence

from explanations import explain_decision
from money import format_amount
from normalization import NormalizedDataset, NormalizedRequest
from planner import Candidate, decide_for_request
from reconstruction import ReconstructedDataset
from validate_output import DecisionRow


def decide_request(
    request: NormalizedRequest,
    normalized: NormalizedDataset,
    reconstructed: ReconstructedDataset,
) -> DecisionRow:
    profile = normalized.profiles_by_user_id[request.user_id]
    state = reconstructed.users[request.user_id]
    options = normalized.payment_options_by_request_id.get(request.request_id, ())
    _forecast, safe_today, earliest, winner = decide_for_request(request, profile, state, options)
    return _row_from_winner(request, state, winner, safe_today, earliest)


def decide_requests(
    normalized: NormalizedDataset,
    reconstructed: ReconstructedDataset,
    requests: Sequence[NormalizedRequest] | None = None,
    limit: int | None = None,
) -> tuple[DecisionRow, ...]:
    target = list(requests if requests is not None else normalized.requests)
    if limit is not None:
        target = target[:limit]
    return tuple(decide_request(request, normalized, reconstructed) for request in target)


def _row_from_winner(
    request: NormalizedRequest,
    state,
    winner: Candidate,
    safe_today,
    earliest,
) -> DecisionRow:
    earliest_out = earliest.isoformat() if earliest is not None else ""
    if winner.status == "affordable_now":
        earliest_out = request.request_date.isoformat()

    plan = "none"
    if winner.payments:
        plan = "|".join(f"{day.isoformat()}:{raw}" for day, _amount, raw in winner.payments)

    changes_out = "none"
    if winner.spending_changes:
        changes_out = "|".join(item.render() for item in winner.spending_changes)

    return DecisionRow(
        request_id=request.request_id,
        amount_safe_to_pay=format_amount(safe_today),
        affordability_status=winner.status,
        recommended_payment_method=winner.method,
        payment_plan=plan,
        earliest_date_for_full_payment=earliest_out,
        spending_changes_needed=changes_out,
        decision_explanation=explain_decision(request, state, winner, safe_today, earliest),
    )

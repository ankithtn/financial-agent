from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from validate_output import DecisionRow, OUTPUT_COLUMNS


def write_output(path: Path, rows: Iterable[DecisionRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "request_id": row.request_id,
                    "amount_safe_to_pay": row.amount_safe_to_pay,
                    "affordability_status": row.affordability_status,
                    "recommended_payment_method": row.recommended_payment_method,
                    "payment_plan": row.payment_plan,
                    "earliest_date_for_full_payment": row.earliest_date_for_full_payment,
                    "spending_changes_needed": row.spending_changes_needed,
                    "decision_explanation": row.decision_explanation,
                }
            )

from __future__ import annotations

import argparse

from dotenv import load_dotenv

from data_loader import dataset_summary, load_dataset
from decision_engine import decide_requests
from normalization import normalize_dataset, normalized_summary
from output_writer import write_output
from reconstruction import reconstruct_dataset, reconstruction_summary
from validate_output import validate_output_file
from evidence import configure_image_llm
from explanations import configure_explainer_llm
from llm_agent import LLMClient, LLMUsageTracker


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Generate Buy or Wait? decisions.")
    parser.add_argument("--limit", type=int, default=None, help="Number of target requests to process.")
    args = parser.parse_args()

    tracker = LLMUsageTracker()
    llm = LLMClient(tracker)
    configure_image_llm(llm)
    configure_explainer_llm(llm)

    bundle = load_dataset()
    summary = dataset_summary(bundle)

    print("Dataset loaded and validated.")
    for key in sorted(summary):
        print(f"{key}: {summary[key]}")

    normalized = normalize_dataset(bundle)
    normalized_counts = normalized_summary(normalized)
    print("Dataset normalized.")
    for key in sorted(normalized_counts):
        print(f"{key}: {normalized_counts[key]}")

    reconstructed = reconstruct_dataset(normalized)
    reconstructed_counts = reconstruction_summary(reconstructed)
    print("Financial state reconstructed.")
    for key in sorted(reconstructed_counts):
        print(f"{key}: {reconstructed_counts[key]}")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when provided.")

    rows = decide_requests(normalized, reconstructed, limit=args.limit)
    output_path = bundle.repo_root / "output.csv"
    write_output(output_path, rows)
    validate_output_file(output_path, bundle.repo_root, expected_request_ids={row.request_id for row in rows})

    evaluation_dir = bundle.repo_root / "code" / "evaluation"
    tracker.write_report(
        evaluation_dir / "usage_report.md",
        enabled=llm.enabled,
        error=None if llm.enabled else "LLM disabled or API key unavailable; deterministic fallbacks were used.",
        request_count=len(rows),
        run_scope="full-dataset" if args.limit is None else "limited",
    )
    tracker.write_transcript(evaluation_dir / "llm_runtime_transcript.jsonl")
    print(f"Wrote and validated {len(rows)} decision row(s) to {output_path}.")
    print(f"LLM enabled: {llm.enabled}; calls: {len(tracker.records)}")


if __name__ == "__main__":
    main()

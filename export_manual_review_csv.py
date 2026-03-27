import argparse
import csv
import json
from pathlib import Path


DEFAULT_OUTPUT = Path("manual_review.csv")
DEFAULT_INPUTS = [
    Path("results/truthfulqa_dev_benchmark.json"),
    Path("results/truthfulqa_holdout_benchmark.json"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export benchmark records into a single CSV for manual or judge-assisted review."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT),
        help="Path to the output CSV file.",
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Optional benchmark JSON files to include. Defaults to the dev and holdout benchmark files.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"Benchmark file must contain a JSON array: {path}")
    return records


def join_refs(values) -> str:
    if not isinstance(values, list):
        return ""
    return " || ".join(str(value).strip() for value in values if str(value).strip())


def split_name_from_path(path: Path) -> str:
    lowered = path.name.lower()
    if "dev" in lowered:
        return "dev"
    if "holdout" in lowered or "test" in lowered:
        return "holdout"
    return path.stem


def export_csv(input_paths: list[Path], output_path: Path) -> int:
    rows = []
    for path in input_paths:
        split_name = split_name_from_path(path)
        records = load_records(path)
        for record_index, record in enumerate(records):
            label_details = record.get("label_details") or {}
            rows.append(
                {
                    "review_id": f"{split_name}_{record_index:03d}",
                    "split": split_name,
                    "source_file": str(path),
                    "record_index": record_index,
                    "question": record.get("q", ""),
                    "model_answer": record.get("model_answer", ""),
                    "auto_label": record.get("label", ""),
                    "auto_label_method": record.get("label_method", ""),
                    "best_correct_reference": label_details.get("best_correct_reference", ""),
                    "best_correct_score": label_details.get("best_correct_score", ""),
                    "best_incorrect_reference": label_details.get("best_incorrect_reference", ""),
                    "best_incorrect_score": label_details.get("best_incorrect_score", ""),
                    "correct_refs": join_refs(record.get("correct_answers")),
                    "incorrect_refs": join_refs(record.get("incorrect_answers")),
                    "truth_false_mean": record.get("truth_vs_false_consensus_mean", ""),
                    "truth_model_mean": record.get("truth_vs_model_consensus_mean", ""),
                    "logit_confidence": record.get("logit_confidence", ""),
                    "manual_label": "",
                    "manual_confidence": "",
                    "manual_notes": "",
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "split",
                "source_file",
                "record_index",
                "question",
                "model_answer",
                "auto_label",
                "auto_label_method",
                "best_correct_reference",
                "best_correct_score",
                "best_incorrect_reference",
                "best_incorrect_score",
                "correct_refs",
                "incorrect_refs",
                "truth_false_mean",
                "truth_model_mean",
                "logit_confidence",
                "manual_label",
                "manual_confidence",
                "manual_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.inputs] if args.inputs else DEFAULT_INPUTS
    output_path = Path(args.out)
    row_count = export_csv(input_paths=input_paths, output_path=output_path)
    print(f"Saved {row_count} review rows to {output_path}")


if __name__ == "__main__":
    main()

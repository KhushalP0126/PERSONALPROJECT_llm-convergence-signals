import argparse
import json
from pathlib import Path

from datasets import load_dataset


DEFAULT_OUTPUT_FILE = Path("data/truthfulqa_pairs.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download TruthfulQA and convert it into a small paired dataset for consensus analysis."
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of TruthfulQA questions to convert.",
    )
    return parser.parse_args()


def first_text(value):
    if isinstance(value, str):
        text = value.strip()
        return text or None

    if isinstance(value, (list, tuple)):
        for item in value:
            text = first_text(item)
            if text:
                return text

    if isinstance(value, dict):
        for key in ("text", "answer", "best_answer"):
            if key in value:
                text = first_text(value[key])
                if text:
                    return text

    return None


def build_records(limit: int) -> list[dict]:
    dataset = load_dataset("truthful_qa", "generation")
    validation_split = dataset["validation"]

    records = []
    for item in validation_split:
        question = first_text(item.get("question"))
        correct_answer = first_text(item.get("correct_answers")) or first_text(item.get("best_answer"))
        incorrect_answer = first_text(item.get("incorrect_answers"))

        if not question or not correct_answer or not incorrect_answer:
            continue

        records.append(
            {
                "q": question,
                "correct_answer": correct_answer,
                "incorrect_answer": incorrect_answer,
                "source": "truthfulqa_generation_validation",
            }
        )

        if len(records) >= limit:
            break

    return records


def main() -> None:
    args = parse_args()
    records = build_records(limit=args.limit)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(f"Saved {len(records)} TruthfulQA records to {output_path}")


if __name__ == "__main__":
    main()

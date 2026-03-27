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
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="How many prepared examples to skip before starting the slice.",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        help="Optional deterministic shuffle seed. Use the same seed with different offsets to make disjoint slices.",
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


def collect_texts(value) -> list[str]:
    texts: list[str] = []

    if isinstance(value, str):
        text = value.strip()
        if text:
            texts.append(text)
        return texts

    if isinstance(value, (list, tuple)):
        for item in value:
            texts.extend(collect_texts(item))
        return texts

    if isinstance(value, dict):
        for key in ("text", "answer", "best_answer"):
            if key in value:
                texts.extend(collect_texts(value[key]))
        return texts

    return texts


def unique_texts(values) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for text in collect_texts(values):
        normalized = " ".join(text.lower().split())
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(text)
    return deduped


def build_records(limit: int, offset: int = 0, shuffle_seed: int | None = None) -> list[dict]:
    dataset = load_dataset("truthful_qa", "generation")
    validation_split = dataset["validation"]
    if shuffle_seed is not None:
        validation_split = validation_split.shuffle(seed=shuffle_seed)

    records = []
    skipped = 0
    for item in validation_split:
        question = first_text(item.get("question"))
        best_answer = first_text(item.get("best_answer"))
        correct_answers = unique_texts(item.get("correct_answers"))
        incorrect_answers = unique_texts(item.get("incorrect_answers"))

        if best_answer:
            correct_answers = unique_texts([best_answer, *correct_answers])

        correct_answer = best_answer or first_text(correct_answers)
        incorrect_answer = first_text(incorrect_answers)

        if not question or not correct_answer or not incorrect_answer:
            continue

        if skipped < offset:
            skipped += 1
            continue

        records.append(
            {
                "q": question,
                "best_answer": best_answer or correct_answer,
                "correct_answer": correct_answer,
                "correct_answers": correct_answers,
                "incorrect_answer": incorrect_answer,
                "incorrect_answers": incorrect_answers,
                "source": "truthfulqa_generation_validation",
            }
        )

        if len(records) >= limit:
            break

    return records


def main() -> None:
    args = parse_args()
    records = build_records(limit=args.limit, offset=args.offset, shuffle_seed=args.shuffle_seed)
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))
    print(
        f"Saved {len(records)} TruthfulQA records to {output_path} "
        f"(offset={args.offset}, shuffle_seed={args.shuffle_seed})"
    )


if __name__ == "__main__":
    main()

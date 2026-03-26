import argparse
import json
import os
from pathlib import Path

PROJECT_CACHE_DIR = Path(__file__).resolve().parent / ".cache"
PROJECT_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_CACHE_DIR))

MATPLOTLIB_CACHE_DIR = PROJECT_CACHE_DIR / "matplotlib"
MATPLOTLIB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MATPLOTLIB_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")
DEFAULT_OUTPUT_DIR = Path("results/consensus_plots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot layer-wise consensus signals and measure whether conflict separates correct from hallucinated answers."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to a consensus dataset JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where plots and summary files will be written.",
    )
    parser.add_argument(
        "--score-field",
        default="support_scores",
        help="Field containing the per-layer score list.",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Field containing a binary label where 1 means correct/truthful and 0 means hallucinated/wrong.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Sample index to plot individually.",
    )
    parser.add_argument(
        "--conflict-threshold",
        type=float,
        help="Optional manual threshold. If omitted, the script picks a midpoint between class means.",
    )
    parser.add_argument(
        "--overlay-count",
        type=int,
        default=5,
        help="How many sample curves to overlay in the multi-sample figure.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Consensus dataset not found: {path}")

    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Consensus dataset must be a non-empty JSON array.")

    return records


def as_binary_label(value):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and value in {0, 1}:
        return int(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "correct", "truthful"}:
            return 1
        if normalized in {"0", "false", "wrong", "hallucinated"}:
            return 0
    raise ValueError(f"Expected a binary label, got: {value!r}")


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def is_unknown_target(text: str) -> bool:
    normalized = normalize_text(text)
    return normalized in {
        "unknown",
        "unknown_future",
        "i don't know",
        "i dont know",
        "cannot be known yet",
    }


def answer_indicates_unknown(text: str) -> bool:
    normalized = normalize_text(text)
    return any(
        phrase in normalized
        for phrase in (
            "i don't know",
            "i dont know",
            "cannot be known",
            "can't be known",
            "cannot know",
            "unknown",
        )
    )


def derive_binary_label(record: dict) -> int:
    answer = record.get("answer") or record.get("model_answer")
    truth = record.get("gt") or record.get("correct_answer")
    false_answer = record.get("incorrect_answer")

    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("Could not derive a binary label because the record is missing an answer field.")
    if not isinstance(truth, str) or not truth.strip():
        raise ValueError("Could not derive a binary label because the record is missing a ground-truth field.")

    normalized_answer = normalize_text(answer)
    normalized_truth = normalize_text(truth)
    normalized_false = normalize_text(false_answer) if isinstance(false_answer, str) else ""

    if is_unknown_target(truth):
        return 1 if answer_indicates_unknown(answer) else 0

    if normalized_false and normalized_false in normalized_answer and normalized_truth not in normalized_answer:
        return 0
    if normalized_truth in normalized_answer:
        return 1
    if normalized_answer == normalized_truth:
        return 1
    return 0


def conflict_score(scores: list[float]) -> float:
    return float(np.std(scores))


def consensus_mean(scores: list[float]) -> float:
    return float(np.mean(scores))


def positive_layer_fraction(scores: list[float]) -> float:
    return float(np.mean([score > 0 for score in scores]))


def enrich_records(records: list[dict], score_field: str, label_field: str) -> tuple[list[dict], int]:
    enriched = []
    skipped_unlabeled = 0
    for record in records:
        if score_field not in record:
            raise KeyError(f"Missing score field {score_field!r} in record.")
        if label_field not in record:
            raise KeyError(f"Missing label field {label_field!r} in record.")

        scores = record[score_field]
        if not isinstance(scores, list) or not scores:
            raise ValueError(f"Score field {score_field!r} must be a non-empty list.")

        enriched_record = dict(record)
        enriched_record["support_scores"] = scores
        if record[label_field] is None:
            skipped_unlabeled += 1
            continue
        try:
            enriched_record["label"] = as_binary_label(record[label_field])
            enriched_record["label_source"] = label_field
        except ValueError:
            enriched_record["label"] = derive_binary_label(record)
            enriched_record["label_source"] = "derived_from_answer"
        enriched_record["conflict"] = conflict_score(scores)
        enriched_record["consensus_mean"] = float(record.get("consensus_mean", consensus_mean(scores)))
        enriched_record["positive_layer_fraction"] = float(
            record.get("positive_layer_fraction", positive_layer_fraction(scores))
        )
        enriched.append(enriched_record)

    return enriched, skipped_unlabeled


def split_by_label(records: list[dict]) -> tuple[list[dict], list[dict]]:
    correct = [record for record in records if record["label"] == 1]
    wrong = [record for record in records if record["label"] == 0]
    return correct, wrong


def average_scores(records: list[dict]) -> np.ndarray:
    return np.mean([record["support_scores"] for record in records], axis=0)


def choose_threshold(correct: list[dict], wrong: list[dict], manual_threshold: float | None) -> float:
    if manual_threshold is not None:
        return manual_threshold

    if not correct or not wrong:
        return float(np.mean([record["conflict"] for record in correct or wrong]))

    correct_mean = float(np.mean([record["conflict"] for record in correct]))
    wrong_mean = float(np.mean([record["conflict"] for record in wrong]))
    return (correct_mean + wrong_mean) / 2.0


def classifier_accuracy(records: list[dict], threshold: float) -> float:
    matches = 0
    for record in records:
        prediction = 1 if record["conflict"] < threshold else 0
        if prediction == record["label"]:
            matches += 1
    return matches / len(records)


def choose_midpoint_threshold(correct: list[dict], wrong: list[dict], field: str) -> float | None:
    if not correct or not wrong:
        return None
    correct_mean = float(np.mean([record[field] for record in correct]))
    wrong_mean = float(np.mean([record[field] for record in wrong]))
    return (correct_mean + wrong_mean) / 2.0


def classifier_accuracy_by_field(records: list[dict], field: str, threshold: float, higher_is_correct: bool) -> float:
    matches = 0
    for record in records:
        prediction = 1 if record[field] > threshold else 0
        if not higher_is_correct:
            prediction = 1 if record[field] < threshold else 0
        if prediction == record["label"]:
            matches += 1
    return matches / len(records)


def plot_sample(record: dict, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.plot(record["support_scores"], linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(
        f"Sample {record.get('q', 'question')} | "
        f"label={record['label']} | conflict={record['conflict']:.3f}"
    )
    plt.xlabel("Layer")
    plt.ylabel("Support")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_averages(correct: list[dict], wrong: list[dict], output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    if correct:
        plt.plot(average_scores(correct), label="Correct / truthful", linewidth=2)
    if wrong:
        plt.plot(average_scores(wrong), label="Hallucinated / wrong", linewidth=2)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title("Average Layer Consensus")
    plt.xlabel("Layer")
    plt.ylabel("Support")
    if correct or wrong:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_conflict_distribution(correct: list[dict], wrong: list[dict], threshold: float, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    if correct:
        plt.hist([record["conflict"] for record in correct], alpha=0.6, label="Correct / truthful")
    if wrong:
        plt.hist([record["conflict"] for record in wrong], alpha=0.6, label="Hallucinated / wrong")
    plt.axvline(threshold, color="black", linestyle="--", linewidth=1, label=f"threshold={threshold:.3f}")
    plt.title("Conflict Distribution")
    plt.xlabel("Conflict score (std of support curve)")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_overlay(records: list[dict], count: int, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    for record in records[:count]:
        label_name = "correct" if record["label"] == 1 else "wrong"
        plt.plot(record["support_scores"], alpha=0.5, label=label_name)
    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"First {min(count, len(records))} Support Curves")
    plt.xlabel("Layer")
    plt.ylabel("Support")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_summary(
    records: list[dict],
    correct: list[dict],
    wrong: list[dict],
    threshold: float,
    skipped_unlabeled: int,
    output_path: Path,
) -> None:
    consensus_threshold = choose_midpoint_threshold(correct, wrong, field="consensus_mean")
    agreement_threshold = choose_midpoint_threshold(correct, wrong, field="positive_layer_fraction")
    summary = {
        "sample_count": len(records),
        "skipped_unlabeled_count": skipped_unlabeled,
        "correct_count": len(correct),
        "wrong_count": len(wrong),
        "correct_conflict_mean": float(np.mean([record["conflict"] for record in correct])) if correct else None,
        "wrong_conflict_mean": float(np.mean([record["conflict"] for record in wrong])) if wrong else None,
        "threshold": threshold,
        "accuracy": classifier_accuracy(records, threshold) if correct and wrong else None,
        "has_both_classes": bool(correct and wrong),
        "correct_consensus_mean": float(np.mean([record["consensus_mean"] for record in correct])) if correct else None,
        "wrong_consensus_mean": float(np.mean([record["consensus_mean"] for record in wrong])) if wrong else None,
        "consensus_mean_threshold": consensus_threshold,
        "consensus_mean_accuracy": (
            classifier_accuracy_by_field(records, field="consensus_mean", threshold=consensus_threshold, higher_is_correct=True)
            if consensus_threshold is not None
            else None
        ),
        "correct_positive_layer_fraction_mean": (
            float(np.mean([record["positive_layer_fraction"] for record in correct])) if correct else None
        ),
        "wrong_positive_layer_fraction_mean": (
            float(np.mean([record["positive_layer_fraction"] for record in wrong])) if wrong else None
        ),
        "positive_layer_fraction_threshold": agreement_threshold,
        "positive_layer_fraction_accuracy": (
            classifier_accuracy_by_field(
                records,
                field="positive_layer_fraction",
                threshold=agreement_threshold,
                higher_is_correct=True,
            )
            if agreement_threshold is not None
            else None
        ),
    }
    output_path.write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    enriched, skipped_unlabeled = enrich_records(records, score_field=args.score_field, label_field=args.label_field)
    if not enriched:
        raise ValueError("No labeled records were available after skipping unlabeled rows.")
    correct, wrong = split_by_label(enriched)

    sample_index = max(0, min(args.sample_index, len(enriched) - 1))
    threshold = choose_threshold(correct, wrong, args.conflict_threshold)
    accuracy = classifier_accuracy(enriched, threshold) if correct and wrong else None
    agreement_threshold = choose_midpoint_threshold(correct, wrong, field="positive_layer_fraction")
    agreement_accuracy = (
        classifier_accuracy_by_field(
            enriched,
            field="positive_layer_fraction",
            threshold=agreement_threshold,
            higher_is_correct=True,
        )
        if agreement_threshold is not None
        else None
    )

    plot_sample(enriched[sample_index], output_dir / "sample_plot.png")
    plot_averages(correct, wrong, output_dir / "average_support.png")
    plot_conflict_distribution(correct, wrong, threshold, output_dir / "conflict_distribution.png")
    plot_overlay(enriched, args.overlay_count, output_dir / "support_overlay.png")
    save_summary(
        enriched,
        correct,
        wrong,
        threshold,
        skipped_unlabeled,
        output_dir / "summary.json",
    )

    print(f"Saved plots to {output_dir}")
    if skipped_unlabeled:
        print(f"skipped_unlabeled={skipped_unlabeled}")
    if accuracy is None:
        print(
            f"correct={len(correct)} wrong={len(wrong)} "
            f"threshold={threshold:.3f} accuracy=n/a (single class in dataset)"
        )
    else:
        print(
            f"correct={len(correct)} wrong={len(wrong)} "
            f"threshold={threshold:.3f} accuracy={accuracy:.3f}"
        )
    if agreement_accuracy is not None:
        print(
            f"positive_layer_fraction_threshold={agreement_threshold:.3f} "
            f"positive_layer_fraction_accuracy={agreement_accuracy:.3f}"
        )


if __name__ == "__main__":
    main()

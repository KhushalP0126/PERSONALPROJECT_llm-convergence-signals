import argparse
import json
import os
import random
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

from analyze_convergence_metrics import (
    binomial_p_value_greater_equal,
    bootstrap_interval,
    cohens_d,
    common_language_effect_size,
    labeled_records,
    late_slope,
    load_records,
    permutation_p_value,
)


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark_reviewed.json")
DEFAULT_OUTPUT_DIR = Path("results/late_slope_holdout")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a locked holdout test for the late_slope convergence metric."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(DEFAULT_INPUT_FILE),
        help="Development dataset path, or the full dataset to split if --holdout-in is omitted.",
    )
    parser.add_argument(
        "--holdout-in",
        dest="holdout_input_path",
        help="Optional separate holdout dataset path. If omitted, a stratified split is created from --in.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where the holdout evaluation outputs will be written.",
    )
    parser.add_argument(
        "--score-field",
        default="support_scores",
        help="Field containing the layer support scores.",
    )
    parser.add_argument(
        "--label-field",
        default="label",
        help="Field containing binary labels where 1 means correct.",
    )
    parser.add_argument(
        "--late-window",
        type=int,
        default=5,
        help="How many final layers to use for the late_slope metric.",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=0.7,
        help="Development fraction when splitting a single dataset into dev and holdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for the split.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Permutation samples for the holdout effect test.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=10000,
        help="Bootstrap resamples for the holdout confidence interval.",
    )
    return parser.parse_args()


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def stratified_split_indices(labels: np.ndarray, dev_fraction: float, seed: int) -> tuple[list[int], list[int]]:
    rng = np.random.default_rng(seed)
    dev_indices: list[int] = []
    holdout_indices: list[int] = []

    for label_value in [1, 0]:
        label_indices = np.where(labels == label_value)[0]
        shuffled = np.array(label_indices, copy=True)
        rng.shuffle(shuffled)

        dev_count = max(1, int(round(len(shuffled) * dev_fraction)))
        dev_count = min(dev_count, len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
        dev_indices.extend(shuffled[:dev_count].tolist())
        holdout_indices.extend(shuffled[dev_count:].tolist())

    dev_indices.sort()
    holdout_indices.sort()
    return dev_indices, holdout_indices


def late_slope_values(records: list[dict], score_field: str, late_window: int) -> np.ndarray:
    return np.asarray(
        [late_slope(np.asarray(record[score_field], dtype=float), late_window=late_window) for record in records],
        dtype=float,
    )


def fixed_threshold(dev_values: np.ndarray, dev_labels: np.ndarray) -> float:
    correct_values = dev_values[dev_labels == 1]
    wrong_values = dev_values[dev_labels == 0]
    return float((correct_values.mean() + wrong_values.mean()) / 2.0)


def plot_split_distributions(
    dev_values: np.ndarray,
    dev_labels: np.ndarray,
    holdout_values: np.ndarray,
    holdout_labels: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for axis, values, labels, title in [
        (axes[0], dev_values, dev_labels, "Development Split"),
        (axes[1], holdout_values, holdout_labels, "Holdout Split"),
    ]:
        axis.hist(values[labels == 1], bins=14, alpha=0.6, label="Correct", color="#4c72b0")
        axis.hist(values[labels == 0], bins=14, alpha=0.6, label="Wrong", color="#c44e52")
        axis.axvline(threshold, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("late_slope")

    axes[0].set_ylabel("Count")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def with_record_indices(records: list[dict]) -> list[dict]:
    enriched = []
    for index, record in enumerate(records):
        enriched_record = dict(record)
        enriched_record["_record_index"] = index
        enriched.append(enriched_record)
    return enriched


def split_records(records: list[dict], label_field: str, dev_fraction: float, seed: int) -> tuple[list[dict], list[dict], dict]:
    labels = np.asarray([int(record[label_field]) for record in records], dtype=int)
    dev_indices, holdout_indices = stratified_split_indices(labels=labels, dev_fraction=dev_fraction, seed=seed)
    dev_records = [records[index] for index in dev_indices]
    holdout_records = [records[index] for index in holdout_indices]
    split_metadata = {
        "mode": "single_dataset_split",
        "dev_fraction": dev_fraction,
        "seed": seed,
        "dev_indices": [int(records[index]["_record_index"]) for index in dev_indices],
        "holdout_indices": [int(records[index]["_record_index"]) for index in holdout_indices],
    }
    return dev_records, holdout_records, split_metadata


def separate_datasets(dev_records: list[dict], holdout_records: list[dict]) -> dict:
    return {
        "mode": "separate_datasets",
        "holdout_is_fresh_dataset": True,
        "dev_indices": [int(record["_record_index"]) for record in dev_records],
        "holdout_indices": [int(record["_record_index"]) for record in holdout_records],
    }


def record_brief(records: list[dict]) -> list[dict]:
    return [
        {
            "record_index": int(record["_record_index"]),
            "label": int(record["label"]),
            "question": record.get("q"),
            "label_method": record.get("label_method"),
        }
        for record in records
    ]


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dev_source = with_record_indices(load_records(input_path))
    dev_labeled, dev_skipped = labeled_records(dev_source, score_field=args.score_field, label_field=args.label_field)

    if args.holdout_input_path:
        holdout_source = with_record_indices(load_records(Path(args.holdout_input_path)))
        holdout_labeled, holdout_skipped = labeled_records(
            holdout_source,
            score_field=args.score_field,
            label_field=args.label_field,
        )
        dev_records = dev_labeled
        holdout_records = holdout_labeled
        split_metadata = separate_datasets(dev_records=dev_records, holdout_records=holdout_records)
        split_metadata["dev_input_path"] = str(input_path)
        split_metadata["holdout_input_path"] = str(Path(args.holdout_input_path))
        split_metadata["dev_skipped_unlabeled_count"] = dev_skipped
        split_metadata["holdout_skipped_unlabeled_count"] = holdout_skipped
    else:
        dev_records, holdout_records, split_metadata = split_records(
            records=dev_labeled,
            label_field=args.label_field,
            dev_fraction=args.dev_fraction,
            seed=args.seed,
        )
        split_metadata["input_path"] = str(input_path)
        split_metadata["skipped_unlabeled_count"] = dev_skipped
        split_metadata["holdout_is_fresh_dataset"] = False

    dev_labels = np.asarray([int(record[args.label_field]) for record in dev_records], dtype=int)
    holdout_labels = np.asarray([int(record[args.label_field]) for record in holdout_records], dtype=int)
    dev_values = late_slope_values(records=dev_records, score_field=args.score_field, late_window=args.late_window)
    holdout_values = late_slope_values(records=holdout_records, score_field=args.score_field, late_window=args.late_window)

    threshold = fixed_threshold(dev_values=dev_values, dev_labels=dev_labels)

    def evaluated(values: np.ndarray, labels: np.ndarray, seed: int) -> dict:
        correct_values = values[labels == 1]
        wrong_values = values[labels == 0]
        predictions = (values > threshold).astype(int)
        accuracy = float(np.mean(predictions == labels))
        correct_count = int(np.sum(predictions == labels))
        return {
            "sample_count": int(len(values)),
            "correct_count": int(np.sum(labels == 1)),
            "wrong_count": int(np.sum(labels == 0)),
            "correct_mean": float(correct_values.mean()),
            "wrong_mean": float(wrong_values.mean()),
            "mean_difference_correct_minus_wrong": float(correct_values.mean() - wrong_values.mean()),
            "cohens_d": cohens_d(correct_values=correct_values, wrong_values=wrong_values),
            "common_language_effect_size": common_language_effect_size(
                correct_values=correct_values,
                wrong_values=wrong_values,
            ),
            "permutation_p_value": permutation_p_value(
                correct_values=correct_values,
                wrong_values=wrong_values,
                permutations=args.permutations,
                rng=random.Random(seed),
            ),
            "bootstrap_95ci_correct_minus_wrong": bootstrap_interval(
                correct_values=correct_values,
                wrong_values=wrong_values,
                bootstraps=args.bootstraps,
                rng=random.Random(seed + 1),
            ),
            "threshold": threshold,
            "threshold_accuracy": accuracy,
            "threshold_correct_count": correct_count,
            "threshold_binomial_p_value": binomial_p_value_greater_equal(
                successes=correct_count,
                trials=len(values),
            ),
        }

    summary = {
        "metric_name": "late_slope",
        "late_window": args.late_window,
        "split": split_metadata,
        "development": evaluated(values=dev_values, labels=dev_labels, seed=args.seed),
        "holdout": evaluated(values=holdout_values, labels=holdout_labels, seed=args.seed + 100),
        "development_records": record_brief(dev_records),
        "holdout_records": record_brief(holdout_records),
        "locked_rule": {
            "metric": "late_slope",
            "definition": "slope of a first-order fit over the final late_window support scores",
            "higher_is_more_truthful": True,
            "threshold_source": "midpoint between development correct and wrong means",
        },
        "caveat": (
            "If split.mode is single_dataset_split, this is only a post-hoc holdout estimate because the same "
            "reviewed dataset informed the metric discovery. Use --holdout-in with a fresh labeled benchmark file "
            "for a cleaner external validation."
        ),
    }

    save_json(summary, output_dir / "summary.json")
    plot_split_distributions(
        dev_values=dev_values,
        dev_labels=dev_labels,
        holdout_values=holdout_values,
        holdout_labels=holdout_labels,
        threshold=threshold,
        output_path=output_dir / "late_slope_holdout.png",
    )

    print(
        f"dev_accuracy={summary['development']['threshold_accuracy']:.3f} "
        f"holdout_accuracy={summary['holdout']['threshold_accuracy']:.3f} "
        f"threshold={threshold:.3f}"
    )
    print(f"Saved holdout evaluation to {output_dir}")


if __name__ == "__main__":
    main()

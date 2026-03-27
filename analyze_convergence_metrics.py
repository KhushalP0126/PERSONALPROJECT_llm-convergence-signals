import argparse
import json
import math
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


DEFAULT_INPUT_FILE = Path("results/truthfulqa_consensus_benchmark_reviewed.json")
DEFAULT_OUTPUT_DIR = Path("results/convergence_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze late-layer convergence signals as an alternative to the failed conflict hypothesis."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to the benchmark JSON file.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where convergence summaries and plots will be written.",
    )
    parser.add_argument(
        "--score-field",
        default="support_scores",
        help="Field containing the truth-vs-false layer scores.",
    )
    parser.add_argument(
        "--truth-model-field",
        default="truth_vs_model_scores",
        help="Field containing truth-vs-model layer scores.",
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
        help="How many final layers to use for the late-layer metrics.",
    )
    parser.add_argument(
        "--early-window",
        type=int,
        default=5,
        help="How many initial layers to use for the early-to-late comparison.",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Permutation samples for the mean-difference test.",
    )
    parser.add_argument(
        "--bootstraps",
        type=int,
        default=10000,
        help="Bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help="Random seed for resampling.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Benchmark file must contain a non-empty JSON array.")
    return records


def labeled_records(records: list[dict], score_field: str, label_field: str) -> tuple[list[dict], int]:
    labeled: list[dict] = []
    skipped = 0
    for record in records:
        if record.get(label_field) not in {0, 1}:
            skipped += 1
            continue
        scores = record.get(score_field)
        if not isinstance(scores, list) or not scores:
            skipped += 1
            continue
        labeled.append(record)
    if not labeled:
        raise ValueError("No labeled records were available for convergence analysis.")
    return labeled, skipped


def pooled_std(correct_values: np.ndarray, wrong_values: np.ndarray) -> float:
    if len(correct_values) < 2 or len(wrong_values) < 2:
        return 0.0
    numerator = ((len(correct_values) - 1) * correct_values.var(ddof=1)) + (
        (len(wrong_values) - 1) * wrong_values.var(ddof=1)
    )
    denominator = len(correct_values) + len(wrong_values) - 2
    if denominator <= 0:
        return 0.0
    return float(math.sqrt(max(numerator / denominator, 0.0)))


def cohens_d(correct_values: np.ndarray, wrong_values: np.ndarray) -> float | None:
    scale = pooled_std(correct_values=correct_values, wrong_values=wrong_values)
    if scale == 0:
        return None
    return float((correct_values.mean() - wrong_values.mean()) / scale)


def common_language_effect_size(correct_values: np.ndarray, wrong_values: np.ndarray) -> float | None:
    if len(correct_values) == 0 or len(wrong_values) == 0:
        return None
    favorable = 0.0
    total = 0
    for correct_value in correct_values:
        for wrong_value in wrong_values:
            total += 1
            if correct_value > wrong_value:
                favorable += 1.0
            elif correct_value == wrong_value:
                favorable += 0.5
    if total == 0:
        return None
    return favorable / total


def permutation_p_value(
    correct_values: np.ndarray,
    wrong_values: np.ndarray,
    permutations: int,
    rng: random.Random,
) -> float:
    observed = float(correct_values.mean() - wrong_values.mean())
    combined = np.concatenate([correct_values, wrong_values])
    correct_count = len(correct_values)
    extreme = 0
    for _ in range(permutations):
        indices = list(range(len(combined)))
        rng.shuffle(indices)
        shuffled = combined[indices]
        perm_correct = shuffled[:correct_count]
        perm_wrong = shuffled[correct_count:]
        perm_diff = float(perm_correct.mean() - perm_wrong.mean())
        if perm_diff >= observed:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


def bootstrap_interval(
    correct_values: np.ndarray,
    wrong_values: np.ndarray,
    bootstraps: int,
    rng: random.Random,
) -> tuple[float, float]:
    diffs: list[float] = []
    for _ in range(bootstraps):
        resampled_correct = np.asarray(
            [correct_values[rng.randrange(len(correct_values))] for _ in range(len(correct_values))],
            dtype=float,
        )
        resampled_wrong = np.asarray(
            [wrong_values[rng.randrange(len(wrong_values))] for _ in range(len(wrong_values))],
            dtype=float,
        )
        diffs.append(float(resampled_correct.mean() - resampled_wrong.mean()))
    lower, upper = np.percentile(diffs, [2.5, 97.5])
    return float(lower), float(upper)


def binomial_p_value_greater_equal(successes: int, trials: int, baseline: float = 0.5) -> float:
    if trials <= 0:
        return 1.0
    probability = 0.0
    for count in range(successes, trials + 1):
        probability += math.comb(trials, count) * (baseline ** count) * ((1.0 - baseline) ** (trials - count))
    return probability


def fixed_threshold_accuracy(values: np.ndarray, labels: np.ndarray, higher_is_correct: bool) -> tuple[float, float, int]:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    threshold = float((correct_values.mean() + wrong_values.mean()) / 2.0)
    if higher_is_correct:
        predictions = (values > threshold).astype(int)
    else:
        predictions = (values < threshold).astype(int)
    correct_count = int(np.sum(predictions == labels))
    return float(correct_count / len(labels)), threshold, correct_count


def late_slope(scores: np.ndarray, late_window: int) -> float:
    tail = scores[-late_window:] if len(scores) >= late_window else scores
    x = np.arange(len(tail), dtype=float)
    if len(tail) <= 1:
        return 0.0
    return float(np.polyfit(x, tail, 1)[0])


def extract_metric_values(
    records: list[dict],
    score_field: str,
    truth_model_field: str,
    early_window: int,
    late_window: int,
) -> dict[str, np.ndarray]:
    support_curves = [np.asarray(record[score_field], dtype=float) for record in records]
    truth_model_curves = [
        np.asarray(record.get(truth_model_field, record[score_field]), dtype=float) for record in records
    ]

    return {
        "late_slope": np.asarray([late_slope(curve, late_window=late_window) for curve in support_curves], dtype=float),
        "final_support": np.asarray([curve[-1] for curve in support_curves], dtype=float),
        "late_mean": np.asarray(
            [float(np.mean(curve[-late_window:] if len(curve) >= late_window else curve)) for curve in support_curves],
            dtype=float,
        ),
        "early_to_late_delta": np.asarray(
            [
                float(
                    np.mean(curve[-late_window:] if len(curve) >= late_window else curve)
                    - np.mean(curve[:early_window] if len(curve) >= early_window else curve)
                )
                for curve in support_curves
            ],
            dtype=float,
        ),
        "truth_model_final": np.asarray([curve[-1] for curve in truth_model_curves], dtype=float),
        "overall_abs_mean": np.asarray([float(np.mean(np.abs(curve))) for curve in support_curves], dtype=float),
    }


def metric_summary(
    name: str,
    values: np.ndarray,
    labels: np.ndarray,
    permutations: int,
    bootstraps: int,
    rng_seed: int,
) -> dict:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    accuracy, threshold, threshold_correct = fixed_threshold_accuracy(
        values=values,
        labels=labels,
        higher_is_correct=True,
    )
    rng_perm = random.Random(rng_seed)
    rng_boot = random.Random(rng_seed + 1)

    return {
        "metric_name": name,
        "sample_count": int(len(values)),
        "correct_count": int(len(correct_values)),
        "wrong_count": int(len(wrong_values)),
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
            permutations=permutations,
            rng=rng_perm,
        ),
        "bootstrap_95ci_correct_minus_wrong": bootstrap_interval(
            correct_values=correct_values,
            wrong_values=wrong_values,
            bootstraps=bootstraps,
            rng=rng_boot,
        ),
        "threshold": threshold,
        "threshold_accuracy": accuracy,
        "threshold_correct_count": threshold_correct,
        "threshold_binomial_p_value": binomial_p_value_greater_equal(
            successes=threshold_correct,
            trials=len(values),
        ),
        "higher_is_correct": True,
    }


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def plot_accuracy_bars(summary: dict, output_path: Path) -> None:
    metric_entries = list(summary["metrics"].values())
    labels = [entry["metric_name"] for entry in metric_entries]
    accuracies = [entry["threshold_accuracy"] for entry in metric_entries]

    plt.figure(figsize=(10, 5))
    colors = ["#4c72b0" if accuracy >= 0.6 else "#c44e52" for accuracy in accuracies]
    plt.bar(labels, accuracies, color=colors)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1)
    plt.ylabel("Threshold Accuracy")
    plt.title("Convergence Metrics on Reviewed TruthfulQA Benchmark")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_distribution(values: np.ndarray, labels: np.ndarray, metric_name: str, output_path: Path) -> None:
    correct_values = values[labels == 1]
    wrong_values = values[labels == 0]
    plt.figure(figsize=(9, 4))
    plt.hist(correct_values, bins=18, alpha=0.6, label="Correct", color="#4c72b0")
    plt.hist(wrong_values, bins=18, alpha=0.6, label="Wrong", color="#c44e52")
    plt.title(f"{metric_name} Distribution")
    plt.xlabel(metric_name)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    labeled, skipped = labeled_records(records=records, score_field=args.score_field, label_field=args.label_field)
    labels = np.asarray([int(record[args.label_field]) for record in labeled], dtype=int)
    metric_values = extract_metric_values(
        records=labeled,
        score_field=args.score_field,
        truth_model_field=args.truth_model_field,
        early_window=args.early_window,
        late_window=args.late_window,
    )

    summary = {
        "input_path": str(input_path),
        "sample_count": len(records),
        "labeled_count": len(labeled),
        "skipped_unlabeled_count": skipped,
        "late_window": args.late_window,
        "early_window": args.early_window,
        "metrics": {},
    }

    for offset, (metric_name, values) in enumerate(metric_values.items()):
        summary["metrics"][metric_name] = metric_summary(
            name=metric_name,
            values=values,
            labels=labels,
            permutations=args.permutations,
            bootstraps=args.bootstraps,
            rng_seed=args.seed + offset * 100,
        )

    save_json(summary, output_dir / "summary.json")
    plot_accuracy_bars(summary=summary, output_path=output_dir / "metric_accuracy.png")
    for metric_name, values in metric_values.items():
        plot_distribution(
            values=values,
            labels=labels,
            metric_name=metric_name,
            output_path=output_dir / f"{metric_name}_distribution.png",
        )

    ranked = sorted(
        summary["metrics"].values(),
        key=lambda entry: entry["threshold_accuracy"],
        reverse=True,
    )
    top_metric = ranked[0]
    print(
        f"labeled={len(labeled)} skipped_unlabeled={skipped} "
        f"best_metric={top_metric['metric_name']} accuracy={top_metric['threshold_accuracy']:.3f} "
        f"p={top_metric['permutation_p_value']:.4f}"
    )
    print(f"Saved convergence analysis to {output_dir}")


if __name__ == "__main__":
    main()

import argparse
import json
from pathlib import Path


DEFAULT_INPUT_FILE = Path("results/consensus_dataset.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize which layers support the target token and whether they recur in the same regions."
    )
    parser.add_argument(
        "--in",
        dest="input_path",
        default=str(DEFAULT_INPUT_FILE),
        help="Path to a consensus dataset JSON file.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Consensus dataset not found: {path}")

    records = json.loads(path.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError("Consensus dataset must be a non-empty JSON array.")

    return records


def region_name(layer_index: int, layer_count: int) -> str:
    third = max(1, layer_count // 3)
    if layer_index < third:
        return "early"
    if layer_index < 2 * third:
        return "middle"
    return "late"


def region_histogram(indices: list[int], layer_count: int) -> dict:
    histogram = {"early": 0, "middle": 0, "late": 0}
    for index in indices:
        histogram[region_name(index, layer_count)] += 1
    return histogram


def shared_positive_frequencies(records: list[dict]) -> list[float]:
    layer_count = records[0]["layer_count"]
    counts = [0] * layer_count

    for record in records:
        for index in record["positive_layer_indices"]:
            counts[index] += 1

    total = len(records)
    return [count / total for count in counts]


def print_per_example(records: list[dict]) -> None:
    for record in records:
        region_counts = region_histogram(
            indices=record["positive_layer_indices"],
            layer_count=record["layer_count"],
        )
        print(record["q"])
        print(
            f"  positive layers: {record['positive_layer_count']}/{record['layer_count']} "
            f"({record['positive_layer_fraction']:.2f})"
        )
        print(f"  positive indices: {record['positive_layer_indices']}")
        print(f"  positive ranges: {record['positive_layer_ranges']}")
        print(f"  positive regions: {region_counts}")
        print(
            f"  strongest support: layer {record['strongest_support_layer']} "
            f"({record['strongest_support_score']:.3f})"
        )
        print(
            f"  strongest opposition: layer {record['strongest_opposition_layer']} "
            f"({record['strongest_opposition_score']:.3f})"
        )
        print()


def print_shared_summary(records: list[dict]) -> None:
    frequencies = shared_positive_frequencies(records)
    recurring_layers = [
        index
        for index, frequency in enumerate(frequencies)
        if frequency > 0
    ]
    dominant_layers = [
        index
        for index, frequency in enumerate(frequencies)
        if frequency >= 0.5
    ]

    print("Shared support summary")
    print(f"  recurring positive layers: {recurring_layers}")
    print(f"  positive in at least half the examples: {dominant_layers}")
    print("  positive frequency by layer:")
    print(f"    {frequencies}")


def main() -> None:
    args = parse_args()
    records = load_records(Path(args.input_path))
    print_per_example(records)
    print_shared_summary(records)


if __name__ == "__main__":
    main()

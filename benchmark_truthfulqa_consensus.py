import argparse
import json
from pathlib import Path

import torch

from build_consensus_dataset import (
    DEFAULT_MODEL,
    PHI2_MODEL,
    build_consensus_prompt,
    choose_comparison_token,
    generate_raw_text,
    get_last_token_logits,
    layer_support_scores,
    layer_logits,
    load_model,
    resolve_device,
    summarize_layer_scores,
    validate_device,
)


DEFAULT_DATASET = Path("data/truthfulqa_pairs.json")
DEFAULT_OUTPUT_FILE = Path("results/truthfulqa_consensus_benchmark.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the consensus pipeline on a prepared TruthfulQA paired dataset."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to load. Use {PHI2_MODEL} if you want to try Phi-2.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to the prepared TruthfulQA JSON file.",
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
        help="Maximum number of prepared records to process.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum tokens to generate for the stored model answer.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Answer-generation temperature. Use 0 for deterministic outputs.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "mps", "cuda", "cpu"],
        default="auto",
        help="Device to use. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--allow-remote-code",
        action="store_true",
        help="Allow custom model code when the selected model requires it.",
    )
    return parser.parse_args()


def load_truthfulqa_records(path: Path, limit: int) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Prepared TruthfulQA dataset not found: {path}. Run prepare_truthfulqa_dataset.py first."
        )

    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError("Prepared TruthfulQA dataset must be a JSON array.")

    filtered = []
    for record in records:
        if "q" not in record or "correct_answer" not in record or "incorrect_answer" not in record:
            raise ValueError("Each TruthfulQA record must contain q, correct_answer, and incorrect_answer.")
        filtered.append(record)
        if len(filtered) >= limit:
            break

    return filtered


def prefixed_summary(scores: list[float], prefix: str) -> dict:
    summary = summarize_layer_scores(scores)
    return {f"{prefix}_{key}": value for key, value in summary.items()}


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def binary_label(model_answer: str, correct_answer: str, incorrect_answer: str, truth_id: int, false_id: int, final_logits):
    normalized_model = normalize_text(model_answer)
    normalized_correct = normalize_text(correct_answer)
    normalized_incorrect = normalize_text(incorrect_answer)

    if normalized_correct in normalized_model and normalized_incorrect not in normalized_model:
        return 1, "answer_match_correct"
    if normalized_incorrect in normalized_model and normalized_correct not in normalized_model:
        return 0, "answer_match_incorrect"

    truth_score = final_logits[0, truth_id].item()
    false_score = final_logits[0, false_id].item()
    return (1 if truth_score > false_score else 0), "truth_false_token_preference"


def raw_token_ids(text: str, tokenizer) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if token_ids:
        return token_ids

    token_ids = tokenizer.encode(f" {text}", add_special_tokens=False)
    if token_ids:
        return token_ids

    raise ValueError(f"Could not derive token ids from text: {text!r}")


def divergence_view(prompt: str, correct_answer: str, incorrect_answer: str, tokenizer, model, device: str):
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    correct_ids = raw_token_ids(correct_answer, tokenizer)
    incorrect_ids = raw_token_ids(incorrect_answer, tokenizer)

    shared_prefix_length = 0
    max_shared = min(len(correct_ids), len(incorrect_ids))
    while (
        shared_prefix_length < max_shared
        and correct_ids[shared_prefix_length] == incorrect_ids[shared_prefix_length]
    ):
        shared_prefix_length += 1

    if shared_prefix_length >= len(correct_ids) or shared_prefix_length >= len(incorrect_ids):
        raise ValueError(
            "TruthfulQA pair does not diverge at a token boundary that this script can analyze."
        )

    analysis_ids = prompt_ids + correct_ids[:shared_prefix_length]
    input_ids = torch.tensor([analysis_ids], device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    return {
        "logits": outputs.logits,
        "hidden_states": outputs.hidden_states,
        "truth_token_id": correct_ids[shared_prefix_length],
        "false_token_id": incorrect_ids[shared_prefix_length],
        "shared_prefix_text": tokenizer.decode(correct_ids[:shared_prefix_length]).strip(),
        "shared_prefix_length": shared_prefix_length,
    }


def build_record(item: dict, tokenizer, model, device: str, max_new_tokens: int, temperature: float) -> dict:
    prompt = build_consensus_prompt(item["q"])
    divergence = divergence_view(
        prompt=prompt,
        correct_answer=item["correct_answer"],
        incorrect_answer=item["incorrect_answer"],
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    logits = divergence["logits"]
    hidden_states = divergence["hidden_states"]
    final_logits = get_last_token_logits(logits)

    truth_id = divergence["truth_token_id"]
    false_id = divergence["false_token_id"]
    predicted_id, model_comparison_id, model_comparison_mode = choose_comparison_token(
        final_logits=final_logits,
        correct_id=truth_id,
    )

    per_layer_logits = layer_logits(hidden_states=hidden_states, model=model)
    truth_vs_false_scores = layer_support_scores(
        layer_logit_outputs=per_layer_logits,
        correct_id=truth_id,
        comparison_id=false_id,
    )
    truth_vs_model_scores = layer_support_scores(
        layer_logit_outputs=per_layer_logits,
        correct_id=truth_id,
        comparison_id=model_comparison_id,
    )
    answer = generate_raw_text(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    label, label_method = binary_label(
        model_answer=answer,
        correct_answer=item["correct_answer"],
        incorrect_answer=item["incorrect_answer"],
        truth_id=truth_id,
        false_id=false_id,
        final_logits=final_logits,
    )

    return {
        "q": item["q"],
        "analysis_prompt": prompt,
        "shared_prefix_text": divergence["shared_prefix_text"],
        "shared_prefix_length": divergence["shared_prefix_length"],
        "source": item.get("source", "truthfulqa"),
        "label": label,
        "label_method": label_method,
        "correct_answer": item["correct_answer"],
        "incorrect_answer": item["incorrect_answer"],
        "model_answer": answer,
        "predicted_token_id": predicted_id,
        "predicted_token": tokenizer.decode([predicted_id]),
        "truth_token_id": truth_id,
        "truth_token": tokenizer.decode([truth_id]),
        "false_token_id": false_id,
        "false_token": tokenizer.decode([false_id]),
        "model_comparison_token_id": model_comparison_id,
        "model_comparison_token": tokenizer.decode([model_comparison_id]),
        "model_comparison_mode": model_comparison_mode,
        "support_scores": truth_vs_false_scores,
        "truth_vs_false_scores": truth_vs_false_scores,
        "truth_vs_false_consensus_mean": sum(truth_vs_false_scores) / len(truth_vs_false_scores),
        "truth_vs_model_scores": truth_vs_model_scores,
        "truth_vs_model_consensus_mean": sum(truth_vs_model_scores) / len(truth_vs_model_scores),
        **prefixed_summary(truth_vs_false_scores, prefix="truth_false"),
        **prefixed_summary(truth_vs_model_scores, prefix="truth_model"),
    }


def save_records(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))


def main() -> None:
    args = parse_args()
    device = resolve_device() if args.device == "auto" else args.device
    validate_device(device)

    records = load_truthfulqa_records(Path(args.dataset), limit=args.limit)
    print(f"Loading {args.model} on {device}...")
    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
        allow_remote_code=args.allow_remote_code,
    )

    output_records = []
    for index, item in enumerate(records, start=1):
        record = build_record(
            item=item,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        output_records.append(record)
        print(
            f"[{index}/{len(records)}] "
            f"truth_false_mean={record['truth_vs_false_consensus_mean']:.3f} "
            f"truth_model_mean={record['truth_vs_model_consensus_mean']:.3f} "
            f"question={record['q']}"
        )

    output_path = Path(args.out)
    save_records(records=output_records, output_path=output_path)
    print(f"Saved {len(output_records)} TruthfulQA consensus records to {output_path}")


if __name__ == "__main__":
    main()

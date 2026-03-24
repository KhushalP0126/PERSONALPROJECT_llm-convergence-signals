import argparse
import json
from pathlib import Path

import torch

from hf_local import (
    DEFAULT_MODEL,
    PHI2_MODEL,
    load_model,
    resolve_device,
    validate_device,
)


DEFAULT_BASE_DATASET = Path("data/base_dataset.json")
DEFAULT_OUTPUT_FILE = Path("results/consensus_dataset.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a layer-wise consensus dataset using a logit-lens style projection."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to load. Use {PHI2_MODEL} if you want to try Phi-2.",
    )
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_BASE_DATASET),
        help="Path to the base JSON dataset containing q, a, and label fields.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to the JSON file where the consensus dataset will be written.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Maximum tokens to generate for the full answer that is stored alongside the signals.",
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


def load_base_dataset(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Base dataset not found: {path}")

    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError("Base dataset must be a JSON array.")

    for record in records:
        if "q" not in record or "a" not in record or "label" not in record:
            raise ValueError("Each dataset item must contain q, a, and label.")

    return records


def build_consensus_prompt(question: str) -> str:
    return (
        "Answer with only the shortest direct answer.\n"
        "If the answer is unknown or cannot be known yet, answer exactly: I don't know.\n"
        f"Question: {question}\n"
        "Answer:"
    )


def tokenize_raw_prompt(prompt: str, tokenizer, device: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {key: value.to(device) for key, value in inputs.items()}


def generate_raw_text(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    temperature: float,
):
    inputs = tokenize_raw_prompt(prompt=prompt, tokenizer=tokenizer, device=device)
    do_sample = temperature > 0
    generation_kwargs = {
        "do_sample": do_sample,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    with torch.inference_mode():
        sequence_ids = model.generate(
            **inputs,
            **generation_kwargs,
        )

    prompt_length = inputs["input_ids"].shape[-1]
    generated_ids = sequence_ids[0][prompt_length:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line or text


def forward_with_layers(prompt: str, tokenizer, model, device: str):
    inputs = tokenize_raw_prompt(prompt=prompt, tokenizer=tokenizer, device=device)
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
    return outputs.logits, outputs.hidden_states


def get_last_token_logits(logits):
    return logits[:, -1, :]


def apply_final_norm(model, hidden_state):
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm(hidden_state)
    if hasattr(model, "transformer") and hasattr(model.transformer, "ln_f"):
        return model.transformer.ln_f(hidden_state)
    if hasattr(model, "model") and hasattr(model.model, "final_layernorm"):
        return model.model.final_layernorm(hidden_state)
    if hasattr(model, "final_layernorm"):
        return model.final_layernorm(hidden_state)
    return hidden_state


def layer_logits(hidden_states, model):
    layer_outputs = []

    # Skip the embedding output and keep only transformer layers.
    for hidden_state in hidden_states[1:]:
        normalized_hidden = apply_final_norm(model=model, hidden_state=hidden_state)
        logits = model.lm_head(normalized_hidden[:, -1, :])
        layer_outputs.append(logits)

    return layer_outputs


def get_candidate_token_ids(text: str, tokenizer) -> list[int]:
    candidates: list[int] = []
    for variant in (text, f" {text}"):
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if token_ids:
            candidate = token_ids[0]
            if candidate not in candidates:
                candidates.append(candidate)

    if not candidates:
        raise ValueError(f"Could not derive a token id from target text: {text!r}")

    return candidates


def choose_correct_token_id(target_text: str, tokenizer, final_logits) -> int:
    candidates = get_candidate_token_ids(text=target_text, tokenizer=tokenizer)
    return max(candidates, key=lambda token_id: final_logits[0, token_id].item())


def choose_comparison_token(final_logits, correct_id: int):
    top_ids = final_logits[0].topk(k=2).indices.tolist()
    predicted_id = top_ids[0]
    if predicted_id == correct_id and len(top_ids) > 1:
        return predicted_id, top_ids[1], "top_alternative"
    return predicted_id, predicted_id, "predicted_token"


def layer_support_scores(layer_logit_outputs, correct_id: int, comparison_id: int) -> list[float]:
    scores = []

    for layer_logit in layer_logit_outputs:
        correct_score = layer_logit[0, correct_id].item()
        comparison_score = layer_logit[0, comparison_id].item()
        scores.append(correct_score - comparison_score)

    return scores


def positive_layer_fraction(scores: list[float]) -> float:
    if not scores:
        return 0.0
    positives = sum(score > 0 for score in scores)
    return positives / len(scores)


def layer_indices(scores: list[float], mode: str) -> list[int]:
    if mode == "positive":
        return [index for index, score in enumerate(scores) if score > 0]
    if mode == "negative":
        return [index for index, score in enumerate(scores) if score < 0]
    if mode == "zero":
        return [index for index, score in enumerate(scores) if score == 0]
    raise ValueError(f"Unsupported layer index mode: {mode}")


def contiguous_ranges(indices: list[int]) -> list[str]:
    if not indices:
        return []

    ranges: list[str] = []
    start = indices[0]
    end = indices[0]

    for index in indices[1:]:
        if index == end + 1:
            end = index
            continue

        ranges.append(f"{start}-{end}" if start != end else str(start))
        start = index
        end = index

    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ranges


def summarize_layer_scores(scores: list[float]) -> dict:
    positive_indices = layer_indices(scores=scores, mode="positive")
    negative_indices = layer_indices(scores=scores, mode="negative")
    zero_indices = layer_indices(scores=scores, mode="zero")
    strongest_support_layer = max(range(len(scores)), key=lambda index: scores[index])
    strongest_opposition_layer = min(range(len(scores)), key=lambda index: scores[index])

    return {
        "layer_count": len(scores),
        "positive_layer_count": len(positive_indices),
        "negative_layer_count": len(negative_indices),
        "zero_layer_count": len(zero_indices),
        "positive_layer_indices": positive_indices,
        "negative_layer_indices": negative_indices,
        "zero_layer_indices": zero_indices,
        "positive_layer_ranges": contiguous_ranges(positive_indices),
        "negative_layer_ranges": contiguous_ranges(negative_indices),
        "strongest_support_layer": strongest_support_layer,
        "strongest_support_score": scores[strongest_support_layer],
        "strongest_opposition_layer": strongest_opposition_layer,
        "strongest_opposition_score": scores[strongest_opposition_layer],
    }


def build_record(item: dict, tokenizer, model, device: str, max_new_tokens: int, temperature: float):
    prompt = build_consensus_prompt(item["q"])
    logits, hidden_states = forward_with_layers(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
    )
    final_logits = get_last_token_logits(logits)
    correct_id = choose_correct_token_id(
        target_text=item["a"],
        tokenizer=tokenizer,
        final_logits=final_logits,
    )
    predicted_id, comparison_id, comparison_mode = choose_comparison_token(
        final_logits=final_logits,
        correct_id=correct_id,
    )
    support_scores = layer_support_scores(
        layer_logit_outputs=layer_logits(hidden_states=hidden_states, model=model),
        correct_id=correct_id,
        comparison_id=comparison_id,
    )
    layer_summary = summarize_layer_scores(support_scores)
    answer = generate_raw_text(
        prompt=prompt,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return {
        "q": item["q"],
        "analysis_prompt": prompt,
        "gt": item["a"],
        "label": item["label"],
        "answer": answer,
        "predicted_token_id": predicted_id,
        "predicted_token": tokenizer.decode([predicted_id]),
        "comparison_token_id": comparison_id,
        "comparison_token": tokenizer.decode([comparison_id]),
        "comparison_mode": comparison_mode,
        "correct_token_id": correct_id,
        "correct_token": tokenizer.decode([correct_id]),
        "support_scores": support_scores,
        "consensus_mean": sum(support_scores) / len(support_scores),
        "positive_layer_fraction": positive_layer_fraction(support_scores),
        **layer_summary,
    }


def save_dataset(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2))


def main() -> None:
    args = parse_args()
    device = resolve_device() if args.device == "auto" else args.device
    validate_device(device)

    dataset_path = Path(args.dataset)
    base_data = load_base_dataset(dataset_path)

    print(f"Loading {args.model} on {device}...")
    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
        allow_remote_code=args.allow_remote_code,
    )

    consensus_records = []
    for index, item in enumerate(base_data, start=1):
        record = build_record(
            item=item,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        consensus_records.append(record)
        print(
            f"[{index}/{len(base_data)}] label={record['label']} "
            f"consensus_mean={record['consensus_mean']:.3f} "
            f"positive_layer_fraction={record['positive_layer_fraction']:.2f} "
            f"question={record['q']}"
        )

    output_path = Path(args.out)
    save_dataset(records=consensus_records, output_path=output_path)
    print(f"Saved {len(consensus_records)} records to {output_path}")


if __name__ == "__main__":
    main()

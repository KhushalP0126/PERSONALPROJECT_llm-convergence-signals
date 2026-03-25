import argparse
import json
import re
from pathlib import Path

import torch

from hf_local import (
    DEFAULT_MODEL,
    PHI2_MODEL,
    generate_text,
    load_model,
    resolve_device,
    validate_device,
)


DEFAULT_QUESTIONS_FILE = Path("data/seed_questions.jsonl")
DEFAULT_OUTPUT_FILE = Path("results/scored_hidden_dataset.jsonl")
SCORE_PATTERN = re.compile(r"[-+]?\d*\.?\d+")
UNKNOWN_FUTURE = "unknown_future"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answers, score them, and capture last-layer hidden states."
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name to load. Use {PHI2_MODEL} if you want to try Phi-2.",
    )
    parser.add_argument(
        "--question",
        action="append",
        help="Question to run. Pass multiple times to build a small batch inline.",
    )
    parser.add_argument(
        "--questions-file",
        default=str(DEFAULT_QUESTIONS_FILE),
        help="JSONL or text file containing questions. Ignored if --question is used.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to the JSONL file where results will be written.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum answer tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Answer-generation temperature. Use 0 for deterministic outputs.",
    )
    parser.add_argument(
        "--score-max-new-tokens",
        type=int,
        default=12,
        help="Maximum tokens for the score-only generation pass.",
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


def load_questions(args: argparse.Namespace) -> list[dict]:
    if args.question:
        return [
            {"question": question.strip()}
            for question in args.question
            if question.strip()
        ]

    path = Path(args.questions_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Questions file not found: {path}. Pass --question or create the file first."
        )

    questions: list[dict] = []
    if path.suffix == ".jsonl":
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            question = (record.get("question") or record.get("prompt") or "").strip()
            if question:
                questions.append(record | {"question": question})
    else:
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if line:
                questions.append({"question": line})

    return questions


def trim_trailing_special_tokens(sequence_ids, tokenizer):
    removable_ids = {
        token_id
        for token_id in {tokenizer.eos_token_id, tokenizer.pad_token_id}
        if token_id is not None
    }
    end_index = sequence_ids.shape[-1]

    while end_index > 1 and sequence_ids[0, end_index - 1].item() in removable_ids:
        end_index -= 1

    return sequence_ids[:, :end_index]


def get_last_hidden(sequence_ids, model, tokenizer):
    trimmed_sequence_ids = trim_trailing_special_tokens(sequence_ids, tokenizer)
    attention_mask = torch.ones_like(trimmed_sequence_ids)

    with torch.inference_mode():
        outputs = model(
            input_ids=trimmed_sequence_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

    last_hidden = outputs.hidden_states[-1][:, -1, :]
    return last_hidden[0].detach().float().cpu()


def build_score_prompt(question: str, answer: str) -> str:
    return (
        "You are grading factual accuracy.\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "Return only one number between 0 and 1.\n"
        "1 means factually correct.\n"
        "0 means clearly incorrect, invented, or hallucinated."
    )


def parse_score(score_text: str) -> float:
    match = SCORE_PATTERN.search(score_text)
    if not match:
        return 0.0

    score = float(match.group(0))
    return max(0.0, min(1.0, score))


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", text.lower()).strip()


def score_from_ground_truth(question_record: dict, answer: str):
    ground_truth = question_record.get("ground_truth")
    if not ground_truth:
        return None

    normalized_answer = normalize_text(answer)
    normalized_ground_truth = normalize_text(str(ground_truth))

    if normalized_ground_truth == UNKNOWN_FUTURE:
        uncertainty_markers = [
            "do not know",
            "don't know",
            "cannot know",
            "cant know",
            "unknown",
            "has not happened",
            "hasn't happened",
            "future",
            "not yet happened",
        ]
        is_uncertain = any(marker in normalized_answer for marker in uncertainty_markers)
        score = 1.0 if is_uncertain else 0.0
        return score, "ground_truth_rule", str(ground_truth)

    score = 1.0 if normalized_ground_truth in normalized_answer else 0.0
    return score, "ground_truth_match", str(ground_truth)


def score_answer(
    question_record: dict,
    answer: str,
    tokenizer,
    model,
    device: str,
    score_max_new_tokens: int,
) -> tuple[float, str, str]:
    ground_truth_score = score_from_ground_truth(question_record=question_record, answer=answer)
    if ground_truth_score is not None:
        score, method, evidence = ground_truth_score
        return score, method, evidence

    raw_score = generate_text(
        prompt=build_score_prompt(question_record["question"], answer),
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=score_max_new_tokens,
        temperature=0.0,
    )
    return parse_score(raw_score), "self_judge", raw_score


def ask(
    question_record: dict,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int,
    temperature: float,
    score_max_new_tokens: int,
    model_name: str,
):
    answer, sequence_ids, _ = generate_text(
        prompt=question_record["question"],
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        return_sequence=True,
    )
    hidden = get_last_hidden(sequence_ids=sequence_ids, model=model, tokenizer=tokenizer)
    score, score_method, score_evidence = score_answer(
        question_record=question_record,
        answer=answer,
        tokenizer=tokenizer,
        model=model,
        device=device,
        score_max_new_tokens=score_max_new_tokens,
    )

    return {
        "question": question_record["question"],
        "answer": answer,
        "score": score,
        "score_method": score_method,
        "score_evidence": score_evidence,
        "hidden": hidden.tolist(),
        "hidden_shape": list(hidden.shape),
        "model": model_name,
        "ground_truth": question_record.get("ground_truth"),
        "category": question_record.get("category"),
    }


def save_results(records: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    device = resolve_device() if args.device == "auto" else args.device
    validate_device(device)

    question_records = load_questions(args)
    if not question_records:
        raise ValueError("No questions were loaded. Provide --question or a non-empty questions file.")

    print(f"Loading {args.model} on {device}...")
    tokenizer, model = load_model(
        model_name=args.model,
        device=device,
        allow_remote_code=args.allow_remote_code,
    )

    records = []
    for index, question_record in enumerate(question_records, start=1):
        record = ask(
            question_record=question_record,
            tokenizer=tokenizer,
            model=model,
            device=device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            score_max_new_tokens=args.score_max_new_tokens,
            model_name=args.model,
        )
        records.append(record)
        print(
            f"[{index}/{len(question_records)}] score={record['score']:.2f} "
            f"hidden_shape={tuple(record['hidden_shape'])} "
            f"question={question_record['question']}"
        )

    output_path = Path(args.out)
    save_results(records, output_path)
    print(f"Saved {len(records)} records to {output_path}")


if __name__ == "__main__":
    main()

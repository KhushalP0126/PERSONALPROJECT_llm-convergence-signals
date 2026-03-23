# Local Hugging Face Chat on Apple Silicon

This project gives you a minimal local text-generation setup for an Apple Silicon Mac.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Verify MPS

```bash
python check_mps.py
```

Expected output:

```text
True
```

## Run The Verified Fast Path

The default model is TinyLlama because it loads faster and was verified to generate correctly on Apple Metal.

Single prompt:

```bash
python local_chat.py --prompt "What is the capital of France?" --temperature 0
```

Interactive loop:

```bash
python local_chat.py
```

## Try Phi-2

If you want the original Phi-2 path from your plan:

```bash
python local_chat.py --model microsoft/phi-2 --prompt "What is the capital of France?" --temperature 0
```

## If Generation Is Slow

Reduce output length:

```bash
python local_chat.py --max-new-tokens 50
```

For more stable factual answers, prefer:

```bash
python local_chat.py --temperature 0
```

## Sanity Checks

Try:

- `Capital of Germany?`
- `Why is the sky blue?`
- `Who won the 2030 World Cup?`

## Day 2 Pipeline

Day 2 adds a simple answer -> score -> hidden-state pipeline for later interpretability work.

Run it on the included seed questions:

```bash
python day2_pipeline.py
```

Run it on one question:

```bash
python day2_pipeline.py --question "What is the capital of France?"
```

Results are written to `results/day2_results.jsonl` and include:

- `question`
- `answer`
- `score`
- `score_method`
- `score_evidence`
- `hidden`
- `hidden_shape`

The hidden-state extraction is done with a second forward pass over the final generated token sequence. That is more stable than relying on generation-time hidden-state return shapes across model versions.

The seed question file in `data/day2_questions.jsonl` includes `ground_truth` labels, so the pipeline will use those when available and only fall back to same-model self-judging when labels are missing. That keeps Day 2 useful for dataset building without pretending the self-score is a strong evaluator.

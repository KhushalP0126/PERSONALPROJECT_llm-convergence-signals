# Hallucination Consensus Analyzer

This project studies hallucination as a failure of internal consensus formation inside a local language model.

It includes:

- local chat and prompt testing on Apple Silicon
- hidden-state collection for question/answer runs
- layer-by-layer consensus scoring
- TruthfulQA benchmark preparation and paired truth-vs-false analysis
- visualization and conflict metrics for proving whether the signal separates correct answers from hallucinations

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Check MPS:

```bash
python check_mps.py
```

## Main Scripts

- `local_chat.py`: interactive local generation
- `build_scored_hidden_dataset.py`: generate answers, score them, and save hidden states
- `build_consensus_dataset.py`: build layer-wise support and opposition scores
- `summarize_layer_support.py`: report where supportive layers occur and whether they recur in the same region
- `prepare_truthfulqa_dataset.py`: download and convert TruthfulQA into a paired dataset
- `benchmark_truthfulqa_consensus.py`: compare truthful vs false answer tokens on TruthfulQA
- `visualize_consensus_patterns.py`: plot curves, compute conflict scores, and evaluate a simple threshold baseline

## Makefile Shortcuts

```bash
make setup
make check-mps
make prompt PROMPT="What is the capital of France?"
make build-hidden-dataset
make build-consensus-dataset
make summarize-layer-support
make prepare-truthfulqa
make benchmark-truthfulqa
make visualize-consensus
```

Useful overrides:

```bash
make prompt MODEL="microsoft/phi-2" PROMPT="What is the capital of Germany?"
make benchmark-truthfulqa TRUTHFULQA_LIMIT=20
make visualize-consensus VIS_INPUT=results/truthfulqa_consensus_benchmark.json
```

## Model Runtime

Verified fast path:

```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

Optional:

```python
model_name = "microsoft/phi-2"
```

The shared runtime lives in `hf_local.py`.

## Seed Data

Hidden-state seed questions:

```text
data/seed_questions.jsonl
```

Consensus seed dataset:

```text
data/consensus_seed_dataset.json
```

Prepared TruthfulQA pairs:

```text
data/truthfulqa_pairs.json
```

Generated TruthfulQA files are ignored by git.

## 1. Local Generation

Single prompt:

```bash
python local_chat.py --prompt "What is the capital of France?" --temperature 0
```

Interactive loop:

```bash
python local_chat.py
```

## 2. Build A Scored Hidden-State Dataset

Run on the seed questions:

```bash
python build_scored_hidden_dataset.py
```

Or one question:

```bash
python build_scored_hidden_dataset.py --question "What is the capital of France?"
```

Output:

```text
results/scored_hidden_dataset.jsonl
```

Each record contains:

- `question`
- `answer`
- `score`
- `score_method`
- `score_evidence`
- `hidden`
- `hidden_shape`

## 3. Build A Consensus Dataset

Run:

```bash
python build_consensus_dataset.py
```

Output:

```text
results/consensus_dataset.json
```

Each record includes:

- `support_scores`
- `positive_layer_fraction`
- `positive_layer_indices`
- `positive_layer_ranges`
- `strongest_support_layer`
- `strongest_opposition_layer`

This script measures support as:

```text
logit(correct_token) - logit(comparison_token)
```

When the model already predicts the correct token, the comparison token becomes the strongest alternative token instead of the correct token itself.

## 4. Summarize Where Support Comes From

Run:

```bash
python summarize_layer_support.py --in results/consensus_dataset.json
```

This tells you:

- how many layers are positive
- which exact layer indices are positive
- whether those layers cluster in early, middle, or late regions
- whether the same regions recur across examples

## 5. Prepare TruthfulQA

Download and convert a small benchmark slice:

```bash
python prepare_truthfulqa_dataset.py --limit 50
```

Output:

```text
data/truthfulqa_pairs.json
```

Each record contains:

```json
{
  "q": "...",
  "correct_answer": "...",
  "incorrect_answer": "...",
  "source": "truthfulqa_generation_validation"
}
```

## 6. Run The TruthfulQA Consensus Benchmark

Run:

```bash
python benchmark_truthfulqa_consensus.py --limit 50
```

Output:

```text
results/truthfulqa_consensus_benchmark.json
```

This script compares truthful and false answers at the first token where they diverge. If both answers share a prefix like:

```text
Fortune cookies originated in ...
```

it compares the first differing token, such as:

```text
San vs China
```

The benchmark output includes:

- `support_scores`
- `label`
- `label_method`
- `truth_vs_false_scores`
- `truth_vs_model_scores`
- `shared_prefix_text`

`label=1` means the run was classified as truthful/correct. `label=0` means hallucinated/wrong. The script first tries to label from the generated answer text and falls back to truthful-vs-false token preference when needed.

## 7. Visualize Consensus And Conflict

Run:

```bash
python visualize_consensus_patterns.py --in results/truthfulqa_consensus_benchmark.json
```

Output directory:

```text
results/consensus_plots/
```

The script saves:

- `sample_plot.png`
- `average_support.png`
- `conflict_distribution.png`
- `support_overlay.png`
- `summary.json`

It computes:

```text
conflict = std(support_scores)
```

and evaluates a simple threshold classifier to test whether conflict separates correct/truthful runs from hallucinated/wrong runs.

If the input file already contains a binary `label`, the script uses it directly. If the `label` field is categorical, it falls back to a simple answer-vs-ground-truth heuristic using `answer`/`gt` or `model_answer`/`correct_answer`.

## Day 4 Interpretation

What to look for:

- correct answers: support curves should become more positive or cleaner late in the stack
- hallucinations: support curves stay noisy, cross zero, or remain negative
- higher conflict: more instability across layers
- lower conflict: cleaner internal agreement

If the threshold accuracy is above random, the signal is doing real work.

## Notes

- The consensus pipeline is a layer-level approximation, not exact neuron attribution.
- TruthfulQA benchmarking is slower than the small seed datasets, so start with a small `--limit`.
- The visualization script expects a binary label field. The TruthfulQA benchmark output already provides one.

## Next Step

Once the plots and conflict scores look meaningful, the next natural step is neuron- or head-level attribution inside the most informative layers.

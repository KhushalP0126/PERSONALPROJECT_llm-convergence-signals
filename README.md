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
- `analyze_neuron_contributions.py`: decompose one selected layer into per-neuron support vs opposition contributions
- `train_stability_detector.py`: train a small detector from conflict-oriented benchmark features
- `analyze_conflict_neuron_patterns.py`: compare recurring neuron patterns in high-conflict wrong cases versus low-conflict correct cases
- `review_benchmark_labels.py`: rapidly review ambiguous benchmark outputs and assign manual labels
- `analyze_conflict_statistics.py`: test whether conflict separates correct and wrong answers with effect sizes and p-values
- `analyze_convergence_metrics.py`: test a pivot hypothesis based on late-layer convergence instead of conflict
- `evaluate_late_slope_holdout.py`: run a locked holdout test for the chosen late-slope convergence metric

## Makefile Shortcuts

```bash
make setup
make check-mps
make ask QUESTION="What is the capital of France?"
make hidden-dataset
make consensus
make layers
make truthfulqa LIMIT=20
make benchmark LIMIT=20
make plots INPUT=results/truthfulqa_consensus_benchmark.json
make neurons QUESTION_MATCH="capital of France" LAYER=19
make detector
make conflict-neurons
make label-benchmark
make conflict-stats
make convergence
make holdout
```

Useful overrides:

```bash
make ask MODEL="microsoft/phi-2" QUESTION="What is the capital of Germany?"
make benchmark LIMIT=20
make plots INPUT=results/truthfulqa_consensus_benchmark.json OUTPUT_DIR=results/consensus_plots
make neurons INPUT=results/consensus_dataset.json QUESTION_MATCH="capital of France" LAYER=19 TOP_K=25
make conflict-neurons HIGH_K=2 LOW_K=2
make label-benchmark LIMIT=10
make conflict-stats STATS_INPUT=results/truthfulqa_consensus_benchmark_reviewed.json
make convergence CONVERGENCE_INPUT=results/truthfulqa_consensus_benchmark_reviewed.json
make holdout DEV_FRACTION=0.7
```

Legacy target names like `prompt`, `build-consensus-dataset`, and `analyze-neurons` still work, but the shorter names above are the preferred interface.

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

For a disjoint slice, use `--offset`. For repeatable dev/holdout slices, use the same `--shuffle-seed` with different offsets:

```bash
python prepare_truthfulqa_dataset.py --limit 100 --offset 0 --shuffle-seed 17 --out data/truthfulqa_dev.json
python prepare_truthfulqa_dataset.py --limit 100 --offset 100 --shuffle-seed 17 --out data/truthfulqa_holdout.json
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

## 8. Zoom Into One Layer And Inspect Neurons

Run:

```bash
python analyze_neuron_contributions.py --in results/consensus_dataset.json --sample-index 0
```

Or with Make:

```bash
make neurons QUESTION_MATCH="capital of France"
```

Output directory:

```text
results/neuron_contributions/record_000_layer_XX/
```

The script:

- picks a layer explicitly or chooses one automatically from the support curve
- reconstructs the same analysis input used by the consensus step
- computes each neuron's support-token and comparison-token contribution
- classifies neurons by net effect on the layer support score
- saves plots and a JSON dump of the per-neuron contributions

Each run writes:

- `contribution_histogram.png`
- `top_neurons.png`
- `neuron_contributions.json`

This is still an approximation. The decomposition is done after the final normalization step, so it is a useful linearized view of neuron influence, not a full causal attribution.

## 9. Train An Internal Stability Detector

Run:

```bash
python train_stability_detector.py --in results/truthfulqa_consensus_benchmark.json
```

Or with Make:

```bash
make detector
```

This script extracts a compact feature set from each support curve:

- overall mean
- conflict standard deviation
- late-layer mean
- late-layer conflict
- spread
- sign flips
- logit confidence

It trains a simple logistic regression model with leave-one-out evaluation and saves:

- `results/stability_detector/summary.json`
- `results/stability_detector/predictions.json`
- `results/stability_detector/feature_weights.png`

## 10. Compare High-Conflict And Low-Conflict Neuron Patterns

Run:

```bash
python analyze_conflict_neuron_patterns.py --in results/truthfulqa_consensus_benchmark.json
```

Or with Make:

```bash
make conflict-neurons
```

## 11. Rapidly Review Ambiguous Labels

Run:

```bash
python review_benchmark_labels.py --in results/truthfulqa_consensus_benchmark.json
```

Or with Make:

```bash
make label-benchmark
```

Default output:

```text
results/truthfulqa_consensus_benchmark_reviewed.json
```

This script is meant for scaling to 100+ examples without changing the benchmark logic itself.

It:

- queues only ambiguous records by default
- shows the question, model answer, and top correct/incorrect references
- lets you press `1` for correct, `0` for wrong, `s` to skip, or `q` to stop and save
- resumes from the reviewed output file if you run it again

For a non-interactive preview:

```bash
python review_benchmark_labels.py --preview-only --limit 5
```

## 12. Measure Whether Conflict Is Statistically Real

Run:

```bash
python analyze_conflict_statistics.py --in results/truthfulqa_consensus_benchmark.json
```

Or with Make:

```bash
make conflict-stats
```

Output:

```text
results/conflict_statistics/summary.json
```

The statistical summary reports both `conflict` and `late_conflict`, including:

- class means
- mean difference (`wrong - correct`)
- Cohen's d
- common-language effect size
- permutation-test p-value
- bootstrap 95% confidence interval
- midpoint-threshold accuracy
- binomial p-value for the threshold baseline

This is the Day 8 scaling check: keep the metric simple, then test whether the separation survives with more reviewed labels.

## 13. Pivot To Late-Layer Convergence

If the conflict hypothesis fails replication, run:

```bash
python analyze_convergence_metrics.py --in results/truthfulqa_consensus_benchmark_reviewed.json
```

Or with Make:

```bash
make convergence
```

Output:

```text
results/convergence_metrics/summary.json
```

This script tests a more specific hypothesis:

- truthful answers end with a more positive support margin
- truthful answers rise more strongly in the final layers
- wrong answers fail to converge late

It evaluates:

- `late_slope`
- `final_support`
- `late_mean`
- `early_to_late_delta`
- `truth_model_final`
- `overall_abs_mean`

Each metric is scored with the fixed interpretation `higher is more truthful`, and the summary reports:

- class means
- effect sizes
- permutation p-values
- bootstrap confidence intervals
- simple midpoint-threshold accuracy

This is an exploratory pivot, not a confirmed result. If one of these metrics looks promising, the next step is to validate it on a fresh benchmark slice rather than reuse the same reviewed set.

## 14. Run A Locked Holdout Test

Once you decide to lock `late_slope` as the metric, run:

```bash
python evaluate_late_slope_holdout.py --in results/truthfulqa_consensus_benchmark_reviewed.json
```

Or with Make:

```bash
make holdout
```

Output:

```text
results/late_slope_holdout/summary.json
```

This script:

- fixes the metric to `late_slope`
- keeps the interpretation fixed: higher late slope means more truthful
- fits the decision threshold on the development split only
- evaluates the holdout split without retuning
- saves the exact split membership for reproducibility

By default it performs a stratified `70/30` split on one reviewed file. For a cleaner external validation, point it at a second unseen reviewed file:

```bash
python evaluate_late_slope_holdout.py \
  --in results/reviewed_dev.json \
  --holdout-in results/reviewed_holdout.json
```

If you use a split from one already-reviewed dataset, treat it as a post-hoc estimate. The cleanest test is a fresh holdout file that played no role in the metric discovery.

Recommended external-validation workflow:

```bash
make benchmark LIMIT=100 TRUTHFULQA_PAIRS=data/truthfulqa_dev.json TRUTHFULQA_BENCHMARK_OUT=results/truthfulqa_dev_benchmark.json TRUTHFULQA_OFFSET=0 TRUTHFULQA_SHUFFLE_SEED=17
make benchmark LIMIT=100 TRUTHFULQA_PAIRS=data/truthfulqa_holdout.json TRUTHFULQA_BENCHMARK_OUT=results/truthfulqa_holdout_benchmark.json TRUTHFULQA_OFFSET=100 TRUTHFULQA_SHUFFLE_SEED=17
```

After reviewing labels for both files, run:

```bash
python evaluate_late_slope_holdout.py \
  --in results/truthfulqa_dev_benchmark_reviewed.json \
  --holdout-in results/truthfulqa_holdout_benchmark_reviewed.json
```

This script:

- selects the highest-conflict wrong cases
- selects the lowest-conflict correct cases
- runs neuron attribution on each selected example
- summarizes recurring supporting and opposing neurons across the two groups

It writes:

- per-case neuron outputs under `results/conflict_neuron_patterns/`
- `results/conflict_neuron_patterns/summary.json`
- frequency plots for recurring neurons in each group

## Notes

- The consensus pipeline is a layer-level approximation, not exact neuron attribution.
- TruthfulQA benchmarking is slower than the small seed datasets, so start with a small `--limit`.
- The visualization script expects a binary label field. The TruthfulQA benchmark output already provides one.

## Next Step

Once the plots and conflict scores look meaningful, the next natural step is neuron- or head-level attribution inside the most informative layers.

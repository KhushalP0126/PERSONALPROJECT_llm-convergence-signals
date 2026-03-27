SHELL := /bin/zsh
.DEFAULT_GOAL := help

VENV ?= venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

MODEL ?=
QUESTION ?= What is the capital of France?
PROMPT ?=
TEMPERATURE ?= 0
MAX_NEW_TOKENS ?= 80

SEED_QUESTIONS ?= data/seed_questions.jsonl
SCORED_HIDDEN_OUT ?= results/scored_hidden_dataset.jsonl

CONSENSUS_SEED_DATASET ?= data/consensus_seed_dataset.json
CONSENSUS_DATASET_OUT ?= results/consensus_dataset.json
LAYER_SUPPORT_IN ?= $(CONSENSUS_DATASET_OUT)

TRUTHFULQA_LIMIT ?= 50
TRUTHFULQA_PAIRS ?= data/truthfulqa_pairs.json
TRUTHFULQA_BENCHMARK_OUT ?= results/truthfulqa_consensus_benchmark.json
TRUTHFULQA_OFFSET ?= 0
TRUTHFULQA_SHUFFLE_SEED ?=
EXTERNAL_LIMIT ?= 100
EXTERNAL_SHUFFLE_SEED ?= 17
EXTERNAL_HOLDOUT_OFFSET ?= $(EXTERNAL_LIMIT)
DEV_PAIRS ?= data/truthfulqa_dev.json
DEV_BENCHMARK_OUT ?= results/truthfulqa_dev_benchmark.json
DEV_REVIEWED_OUT ?= results/truthfulqa_dev_benchmark_reviewed.json
TEST_PAIRS ?= data/truthfulqa_holdout.json
TEST_BENCHMARK_OUT ?= results/truthfulqa_holdout_benchmark.json
TEST_REVIEWED_OUT ?= results/truthfulqa_holdout_benchmark_reviewed.json

VIS_INPUT ?= $(TRUTHFULQA_BENCHMARK_OUT)
VIS_OUTPUT_DIR ?= results/consensus_plots

NEURON_INPUT ?= $(CONSENSUS_DATASET_OUT)
NEURON_OUTPUT_DIR ?= results/neuron_contributions
NEURON_SAMPLE_INDEX ?= 0
NEURON_LAYER_MODE ?= auto
NEURON_LAYER ?=
NEURON_TOP_K ?= 20
QUESTION_CONTAINS ?=

DETECTOR_INPUT ?= $(TRUTHFULQA_BENCHMARK_OUT)
DETECTOR_OUTPUT_DIR ?= results/stability_detector
CONFLICT_INPUT ?= $(TRUTHFULQA_BENCHMARK_OUT)
CONFLICT_OUTPUT_DIR ?= results/conflict_neuron_patterns
HIGH_K ?= 3
LOW_K ?= 3
LABEL_INPUT ?= $(TRUTHFULQA_BENCHMARK_OUT)
LABEL_OUTPUT ?= results/truthfulqa_consensus_benchmark_reviewed.json
STATS_INPUT ?= $(TRUTHFULQA_BENCHMARK_OUT)
STATS_OUTPUT_DIR ?= results/conflict_statistics
CONVERGENCE_INPUT ?= results/truthfulqa_consensus_benchmark_reviewed.json
CONVERGENCE_OUTPUT_DIR ?= results/convergence_metrics
HOLDOUT_INPUT ?= results/truthfulqa_consensus_benchmark_reviewed.json
HOLDOUT_SECONDARY_INPUT ?=
HOLDOUT_OUTPUT_DIR ?= results/late_slope_holdout
DEV_FRACTION ?= 0.7

LIMIT ?=
INPUT ?=
OUTPUT_DIR ?=
SAMPLE ?=
LAYER ?=
TOP_K ?=
QUESTION_MATCH ?=

RUN_QUESTION := $(if $(PROMPT),$(PROMPT),$(QUESTION))
RUN_LIMIT := $(if $(LIMIT),$(LIMIT),$(TRUTHFULQA_LIMIT))
RUN_VIS_INPUT := $(if $(INPUT),$(INPUT),$(VIS_INPUT))
RUN_VIS_OUTPUT_DIR := $(if $(OUTPUT_DIR),$(OUTPUT_DIR),$(VIS_OUTPUT_DIR))
RUN_NEURON_INPUT := $(if $(INPUT),$(INPUT),$(NEURON_INPUT))
RUN_NEURON_OUTPUT_DIR := $(if $(OUTPUT_DIR),$(OUTPUT_DIR),$(NEURON_OUTPUT_DIR))
RUN_SAMPLE := $(if $(SAMPLE),$(SAMPLE),$(NEURON_SAMPLE_INDEX))
RUN_LAYER := $(if $(LAYER),$(LAYER),$(NEURON_LAYER))
RUN_TOP_K := $(if $(TOP_K),$(TOP_K),$(NEURON_TOP_K))
RUN_QUESTION_MATCH := $(if $(QUESTION_MATCH),$(QUESTION_MATCH),$(QUESTION_CONTAINS))

.PHONY: help venv install setup ensure-venv check-mps \
	chat ask prompt \
	hidden-dataset hidden-one build-hidden-dataset build-hidden-dataset-one \
	consensus build-consensus-dataset \
	layers summarize-layer-support \
	truthfulqa prepare-truthfulqa \
	benchmark benchmark-truthfulqa \
	plots visualize-consensus \
	neurons analyze-neurons \
	detector train-detector \
	conflict-neurons analyze-conflict-neurons \
	label-benchmark review-labels \
	conflict-stats analyze-conflict-stats \
	convergence analyze-convergence \
	holdout evaluate-holdout \
	dev-benchmark test-benchmark \
	dev-review test-review \
	external-test

help:
	@echo "Recommended targets:"
	@echo "  make setup"
	@echo "  make check-mps"
	@echo "  make ask QUESTION='What is the capital of France?'"
	@echo "  make hidden-dataset"
	@echo "  make consensus"
	@echo "  make layers"
	@echo "  make truthfulqa LIMIT=20"
	@echo "  make benchmark LIMIT=20"
	@echo "  make plots INPUT=results/truthfulqa_consensus_benchmark.json"
	@echo "  make neurons QUESTION_MATCH='capital of France' LAYER=19"
	@echo "  make detector"
	@echo "  make conflict-neurons"
	@echo "  make label-benchmark"
	@echo "  make conflict-stats"
	@echo "  make convergence"
	@echo "  make holdout"
	@echo ""
	@echo "External validation shortcuts:"
	@echo "  make dev-benchmark"
	@echo "  make test-benchmark"
	@echo "  make dev-review"
	@echo "  make test-review"
	@echo "  make external-test"
	@echo ""
	@echo "Legacy aliases still work:"
	@echo "  prompt build-hidden-dataset build-consensus-dataset summarize-layer-support"
	@echo "  prepare-truthfulqa benchmark-truthfulqa visualize-consensus analyze-neurons"
	@echo "  train-detector analyze-conflict-neurons review-labels analyze-conflict-stats analyze-convergence evaluate-holdout"
	@echo ""
	@echo "Useful variables:"
	@echo "  MODEL='microsoft/phi-2'"
	@echo "  QUESTION='What is the capital of Germany?'"
	@echo "  TEMPERATURE=0"
	@echo "  MAX_NEW_TOKENS=80"
	@echo "  LIMIT=50"
	@echo "  INPUT=results/consensus_dataset.json"
	@echo "  OUTPUT_DIR=results/consensus_plots"
	@echo "  SAMPLE=0"
	@echo "  LAYER=19"
	@echo "  TOP_K=20"
	@echo "  HIGH_K=3"
	@echo "  LOW_K=3"
	@echo "  QUESTION_MATCH='capital of France'"
	@echo "  EXTERNAL_LIMIT=100"
	@echo "  EXTERNAL_SHUFFLE_SEED=17"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

setup: install

ensure-venv:
	@test -x "$(PYTHON)" || (echo "Missing $(PYTHON). Run 'make setup' first." && exit 1)

check-mps: ensure-venv
	$(PYTHON) check_mps.py

chat: ensure-venv
	$(PYTHON) local_chat.py $(if $(MODEL),--model "$(MODEL)") --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

ask prompt: ensure-venv
	$(PYTHON) local_chat.py $(if $(MODEL),--model "$(MODEL)") --prompt "$(RUN_QUESTION)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

hidden-dataset build-hidden-dataset: ensure-venv
	$(PYTHON) build_scored_hidden_dataset.py $(if $(MODEL),--model "$(MODEL)") --questions-file "$(SEED_QUESTIONS)" --out "$(SCORED_HIDDEN_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

hidden-one build-hidden-dataset-one: ensure-venv
	$(PYTHON) build_scored_hidden_dataset.py $(if $(MODEL),--model "$(MODEL)") --question "$(RUN_QUESTION)" --out "$(SCORED_HIDDEN_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

consensus build-consensus-dataset: ensure-venv
	$(PYTHON) build_consensus_dataset.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(CONSENSUS_SEED_DATASET)" --out "$(CONSENSUS_DATASET_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

layers summarize-layer-support: ensure-venv
	$(PYTHON) summarize_layer_support.py --in "$(LAYER_SUPPORT_IN)"

truthfulqa prepare-truthfulqa: ensure-venv
	$(PYTHON) prepare_truthfulqa_dataset.py --out "$(TRUTHFULQA_PAIRS)" --limit $(RUN_LIMIT) --offset $(TRUTHFULQA_OFFSET) $(if $(TRUTHFULQA_SHUFFLE_SEED),--shuffle-seed $(TRUTHFULQA_SHUFFLE_SEED))

benchmark benchmark-truthfulqa: truthfulqa ensure-venv
	$(PYTHON) benchmark_truthfulqa_consensus.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(TRUTHFULQA_PAIRS)" --out "$(TRUTHFULQA_BENCHMARK_OUT)" --limit $(RUN_LIMIT) --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

plots visualize-consensus: ensure-venv
	$(PYTHON) visualize_consensus_patterns.py --in "$(RUN_VIS_INPUT)" --out-dir "$(RUN_VIS_OUTPUT_DIR)"

neurons analyze-neurons: ensure-venv
	$(PYTHON) analyze_neuron_contributions.py $(if $(MODEL),--model "$(MODEL)") --in "$(RUN_NEURON_INPUT)" --out-dir "$(RUN_NEURON_OUTPUT_DIR)" --sample-index $(RUN_SAMPLE) --layer-mode "$(NEURON_LAYER_MODE)" --top-k $(RUN_TOP_K) $(if $(RUN_LAYER),--layer $(RUN_LAYER)) $(if $(RUN_QUESTION_MATCH),--question-contains "$(RUN_QUESTION_MATCH)")

detector train-detector: ensure-venv
	$(PYTHON) train_stability_detector.py --in "$(DETECTOR_INPUT)" --out-dir "$(DETECTOR_OUTPUT_DIR)"

conflict-neurons analyze-conflict-neurons: ensure-venv
	$(PYTHON) analyze_conflict_neuron_patterns.py $(if $(MODEL),--model "$(MODEL)") --in "$(CONFLICT_INPUT)" --out-dir "$(CONFLICT_OUTPUT_DIR)" --high-k $(HIGH_K) --low-k $(LOW_K) --layer-mode "$(NEURON_LAYER_MODE)" --top-k-neurons $(RUN_TOP_K) $(if $(RUN_LAYER),--layer $(RUN_LAYER))

label-benchmark review-labels: ensure-venv
	$(PYTHON) review_benchmark_labels.py --in "$(LABEL_INPUT)" --out "$(LABEL_OUTPUT)" $(if $(LIMIT),--limit $(LIMIT))

conflict-stats analyze-conflict-stats: ensure-venv
	$(PYTHON) analyze_conflict_statistics.py --in "$(STATS_INPUT)" --out-dir "$(STATS_OUTPUT_DIR)"

convergence analyze-convergence: ensure-venv
	$(PYTHON) analyze_convergence_metrics.py --in "$(CONVERGENCE_INPUT)" --out-dir "$(CONVERGENCE_OUTPUT_DIR)"

holdout evaluate-holdout: ensure-venv
	$(PYTHON) evaluate_late_slope_holdout.py --in "$(HOLDOUT_INPUT)" --out-dir "$(HOLDOUT_OUTPUT_DIR)" --dev-fraction $(DEV_FRACTION) $(if $(HOLDOUT_SECONDARY_INPUT),--holdout-in "$(HOLDOUT_SECONDARY_INPUT)")

dev-benchmark: ensure-venv
	$(MAKE) benchmark LIMIT=$(EXTERNAL_LIMIT) TRUTHFULQA_PAIRS="$(DEV_PAIRS)" TRUTHFULQA_BENCHMARK_OUT="$(DEV_BENCHMARK_OUT)" TRUTHFULQA_OFFSET=0 TRUTHFULQA_SHUFFLE_SEED=$(EXTERNAL_SHUFFLE_SEED)

test-benchmark: ensure-venv
	$(MAKE) benchmark LIMIT=$(EXTERNAL_LIMIT) TRUTHFULQA_PAIRS="$(TEST_PAIRS)" TRUTHFULQA_BENCHMARK_OUT="$(TEST_BENCHMARK_OUT)" TRUTHFULQA_OFFSET=$(EXTERNAL_HOLDOUT_OFFSET) TRUTHFULQA_SHUFFLE_SEED=$(EXTERNAL_SHUFFLE_SEED)

dev-review: ensure-venv
	$(MAKE) label-benchmark LABEL_INPUT="$(DEV_BENCHMARK_OUT)" LABEL_OUTPUT="$(DEV_REVIEWED_OUT)"

test-review: ensure-venv
	$(MAKE) label-benchmark LABEL_INPUT="$(TEST_BENCHMARK_OUT)" LABEL_OUTPUT="$(TEST_REVIEWED_OUT)"

external-test: ensure-venv
	$(MAKE) holdout HOLDOUT_INPUT="$(DEV_REVIEWED_OUT)" HOLDOUT_SECONDARY_INPUT="$(TEST_REVIEWED_OUT)"

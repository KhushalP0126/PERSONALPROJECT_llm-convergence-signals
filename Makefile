SHELL := /bin/zsh

VENV ?= venv
PYTHON ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip

MODEL ?=
PROMPT ?= What is the capital of France?
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

VIS_INPUT ?= $(TRUTHFULQA_BENCHMARK_OUT)
VIS_OUTPUT_DIR ?= results/consensus_plots

.PHONY: help venv install setup ensure-venv check-mps chat prompt build-hidden-dataset build-hidden-dataset-one build-consensus-dataset summarize-layer-support prepare-truthfulqa benchmark-truthfulqa visualize-consensus

help:
	@echo "Available targets:"
	@echo "  make setup"
	@echo "  make check-mps"
	@echo "  make chat"
	@echo "  make prompt PROMPT='What is the capital of France?'"
	@echo "  make build-hidden-dataset"
	@echo "  make build-hidden-dataset-one PROMPT='What is the capital of France?'"
	@echo "  make build-consensus-dataset"
	@echo "  make summarize-layer-support"
	@echo "  make prepare-truthfulqa"
	@echo "  make benchmark-truthfulqa"
	@echo "  make visualize-consensus"
	@echo ""
	@echo "Optional variables:"
	@echo "  MODEL='microsoft/phi-2'"
	@echo "  TEMPERATURE=0"
	@echo "  MAX_NEW_TOKENS=80"
	@echo "  TRUTHFULQA_LIMIT=50"
	@echo "  VIS_OUTPUT_DIR=results/consensus_plots"

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

prompt: ensure-venv
	$(PYTHON) local_chat.py $(if $(MODEL),--model "$(MODEL)") --prompt "$(PROMPT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

build-hidden-dataset: ensure-venv
	$(PYTHON) build_scored_hidden_dataset.py $(if $(MODEL),--model "$(MODEL)") --questions-file "$(SEED_QUESTIONS)" --out "$(SCORED_HIDDEN_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

build-hidden-dataset-one: ensure-venv
	$(PYTHON) build_scored_hidden_dataset.py $(if $(MODEL),--model "$(MODEL)") --question "$(PROMPT)" --out "$(SCORED_HIDDEN_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

build-consensus-dataset: ensure-venv
	$(PYTHON) build_consensus_dataset.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(CONSENSUS_SEED_DATASET)" --out "$(CONSENSUS_DATASET_OUT)" --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

summarize-layer-support: ensure-venv
	$(PYTHON) summarize_layer_support.py --in "$(LAYER_SUPPORT_IN)"

prepare-truthfulqa: ensure-venv
	$(PYTHON) prepare_truthfulqa_dataset.py --out "$(TRUTHFULQA_PAIRS)" --limit $(TRUTHFULQA_LIMIT)

benchmark-truthfulqa: ensure-venv
	$(PYTHON) benchmark_truthfulqa_consensus.py $(if $(MODEL),--model "$(MODEL)") --dataset "$(TRUTHFULQA_PAIRS)" --out "$(TRUTHFULQA_BENCHMARK_OUT)" --limit $(TRUTHFULQA_LIMIT) --temperature $(TEMPERATURE) --max-new-tokens $(MAX_NEW_TOKENS)

visualize-consensus: ensure-venv
	$(PYTHON) visualize_consensus_patterns.py --in "$(VIS_INPUT)" --out-dir "$(VIS_OUTPUT_DIR)"

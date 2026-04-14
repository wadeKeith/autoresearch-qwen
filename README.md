# autoresearch-qwen-mlx

This branch preserves the Apple Silicon / MPS variant inside the unified `autoresearch-qwen` repository. Use the `main` branch for the NVIDIA / CUDA multi-GPU version.

Autonomous research loop that improves `Qwen3-VL-4B-Instruct` on the official [DocVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) benchmark by iteratively editing training code. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

An AI agent edits `train.py`, trains, evaluates, keeps or discards, and repeats — indefinitely until stopped.

## How It Works

```
prepare.py          Download dataset + model (once)
      |
train.py            Train (LoRA by default, agent edits this file)
      |
evaluate.py         Evaluate on full validation split -> val_score
      |
run_experiment.py   One-command train + eval pipeline
```

The agent's single objective: **maximize `val_score`** (mean ANLS on the full official validation split).

## Quick Start

```bash
# 1. Install dependencies
uv sync

# 2. Download dataset and model
uv run python prepare.py

# 3. Baseline evaluation (no training, full validation)
uv run python evaluate.py --base-only --split validation

# 4. Train + evaluate in one command
uv run python run_experiment.py

# 5. (Optional) Export blind test predictions
uv run python submit_test.py
```

## Contract

### Fixed (do not edit)

| Component | Description |
|-----------|-------------|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Dataset | `HuggingFaceM4/DocumentVQA` official splits |
| Split contract | `train` (all) / `validation` (all) / `test` (blind) |
| Metric | Mean ANLS on full `validation` split |
| Evaluator | `evaluate.py`, `src/` |
| Pipeline | `run_experiment.py` |

### Mutable (agent edits)

Only `train.py`. Everything in that file is fair game: LoRA config, optimizer, scheduler, prompt formatting, batch size, step budget, etc.

## Repo Layout

```
train.py                            Mutable training code (agent edits this)
evaluate.py                         Fixed evaluator
run_experiment.py                   One-command train -> eval pipeline
prepare.py                          Dataset + model downloader
submit_test.py                      Blind test export + submission packaging
check_submission.py                 Standalone submission validator
program.md                          Agent contract (the full experiment protocol)
src/autoresearch_qwen_mlx/          Fixed library code (config, scoring, data loading)
```

## Running the Autoresearch Loop

The full experiment protocol is in **[program.md](program.md)**. That is the file the agent should read. Below is a summary.

### Setup

```bash
uv sync
uv run python prepare.py
git checkout -b autoresearch/<tag>
```

### The Loop

```
LOOP FOREVER:
  1. Edit train.py
  2. git commit
  3. uv run python run_experiment.py > run.log 2>&1
  4. grep "^val_score:" run.log
  5. If improved: keep. If worse: git reset --hard HEAD~1
  6. Log to results.tsv (untracked)
  7. Repeat
```

The agent runs **indefinitely** until manually stopped. See `program.md` for the complete protocol including crash handling, simplicity criterion, and logging format.

### Agent Prompt

To start the autonomous loop, give your AI agent this prompt:

```
Read program.md carefully. Then let's set up a new experiment run and start the loop.
```

The agent will read `program.md`, propose a branch tag, set up, and begin experimenting autonomously.

## MPS / Apple Silicon Notes

- Both training and evaluation run in **float32** on MPS
- Image resolution is capped at ~1M pixels to avoid MPS INT_MAX tensor errors
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically
- LoRA is strongly recommended over full SFT for memory reasons
- One experiment iteration (train + eval) takes ~3-5 hours on MPS

## Validation vs Test

| Split | Answers | Use case |
|-------|---------|----------|
| `train` | Yes | Training data (all examples) |
| `validation` | Yes | Local scoring — val_score (all examples) |
| `test` | No | Blind prediction export for submission |

## Submission

```bash
uv run python submit_test.py
```

Produces `artifacts/docvqa_test_submission.bundle.zip` for external submission.

## Runtime Configuration

Set automatically by `hub.py`:

- `HF_ENDPOINT=https://hf-mirror.com`
- `HF_HUB_DISABLE_XET=1`
- `PYTORCH_ENABLE_MPS_FALLBACK=1`

Local model cache: `Qwen/Qwen3-VL-4B-Instruct/` (populated by `prepare.py`).

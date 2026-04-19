# autoresearch-qwen — Agent Protocol

This is the authoritative experiment protocol. The agent reads this file.

## Scope

You MAY read the entire repository. You MAY edit only `train.py`.

You MUST NOT edit `prepare.py`, `evaluate.py`, `run_experiment.sh`, `analysis.py`, anything under `src/`, or scoring rules / dataset splits.

## Goal

**Maximize `val_score`** (mean ANLS on the full validation split).

## Setup

1. Agree on a run tag (e.g. `apr13`). The branch `autoresearch/<tag>` must not already exist.
2. If the worktree is dirty, create a dedicated experiment branch or worktree first — never use destructive rollback in a shared worktree.
3. `git checkout -b autoresearch/<tag>`
4. `uv sync`
5. `uv run autoresearch-qwen doctor --json` — fix any blockers it reports.
6. If data is missing: `uv run python prepare.py`.
7. Create `results.tsv` if absent (header: `commit	val_score	train_seconds	memory_gb	status	description`).
8. Record a baseline: `uv run python evaluate.py --base-only --split validation`, log it in `results.tsv`.

## Experiment Loop

```
LOOP FOREVER:
  1. Review results.tsv and git log.
  2. Edit train.py with one hypothesis.
  3. git add train.py && git commit -m "exp: <description>"
  4. ./run_experiment.sh | tee run.log
  5. Read artifacts/last_result.json (or parse RESULT_JSON: from stdout).
  6. If error/missing: inspect run.log, fix trivial bugs, or discard bad ideas.
  7. Log result in results.tsv (do NOT commit results.tsv).
  8. If val_score improved: keep the commit.
  9. If not: git reset --hard HEAD~1 (only in a clean experiment branch).
  10. Repeat.
```

### Training Budget

Default: `MAX_STEPS = 2000`. When `TIME_BUDGET <= 0`, `MAX_STEPS` is the active budget; `NUM_TRAIN_EPOCHS` is secondary. Changing either is allowed but keep experiments comparable.

### Probe Runs

For risky changes, probe first: `TIME_BUDGET = 120` or `MAX_STEPS = 200`. Check the training loss trend. Discard immediately if the loss diverges. Mid-training eval uses a random 1/10 validation subset — the final `val_score` always comes from the full split in `evaluate.py`.

### Prompt Sync

`USER_PROMPT_PREFIX`, `ANSWER_STYLE`, and `INCLUDE_QUESTION_TYPE_HINTS` in `train.py` are saved to `artifacts/prompt_config.json` after training. The evaluator reads that file automatically.

## Logging

`results.tsv` — tab-separated, 6 columns, untracked by git:

| Column | Example |
|--------|---------|
| commit | `a1b2c3d` |
| val_score | `0.4102` (or `0.0000` for crashes) |
| train_seconds | `1823.5` (or `0.0`) |
| memory_gb | `42.3` (peak_vram_mb / 1024, or `0.0`) |
| status | `keep` / `discard` / `crash` |
| description | short text |

## Autonomy

**NEVER STOP.** Do not ask "should I keep going?". Continue indefinitely until manually interrupted. If out of ideas, re-read `train.py`, combine near-misses, or try radical changes.

## Crash Handling

- Trivial bug → fix and re-run.
- Fundamentally broken → log as `crash`, revert, move on.
- Hang / timeout → kill and treat as crash.

## Simplicity

Simpler is better at equal score. A tiny improvement that adds ugly complexity is not worth it. Deleting code for equal score is a win.

## Search Space

- LoRA: rank, alpha, dropout, target modules
- Optimizer, learning rate, scheduler, warmup
- Differential LR (`LORA_LR_MULTIPLIER_B`)
- Batch size, gradient accumulation, `MAX_STEPS` / `NUM_TRAIN_EPOCHS`
- Prompt formatting (`USER_PROMPT_PREFIX`, `ANSWER_STYLE`)
- Full SFT vs LoRA; gradient checkpointing

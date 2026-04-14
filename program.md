# autoresearch-qwen-mlx

This repo is an autonomous research loop for `Qwen3-VL-4B` on DocVQA.

## Read First

- `README.md`
- `prepare.py`
- `train.py`
- `evaluate.py`
- `run_experiment.py`
- `program.md`

## Scope

### You MAY edit

- `train.py`

### You MUST NOT edit

- `prepare.py`
- `evaluate.py`
- `run_experiment.py`
- anything under `src/`
- scoring rules or dataset splits

## Goal

**Get the highest `val_score`** (mean ANLS on the full validation split).

All experiments train on all training data and evaluate on all validation data.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr11`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The agent-relevant files are `README.md`, `program.md`, `train.py`, `evaluate.py`, `run_experiment.py`.
4. **Verify data exists**: Check that `artifacts/documentvqa_snapshot/` contains parquet data. If not, tell the human to run `uv run python prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## The Experiment Loop

Each experiment runs `uv run python run_experiment.py > run.log 2>&1`. This trains the model and evaluates on the full validation split.

LOOP FOREVER:

1. Look at the current git state and `results.tsv` to understand where you are.
2. Edit `train.py` with an experimental idea.
3. `git add train.py && git commit -m "exp: <description>"`
4. Run: `uv run python run_experiment.py > run.log 2>&1`
5. Read results: `grep "^val_score:\|^train_seconds:" run.log`
6. If grep output is empty, the run crashed. Run `tail -n 50 run.log` to see the error. If it's a trivial fix (typo, missing import), fix and re-run. If the idea is fundamentally broken, give up on it.
7. Record the result in `results.tsv` (do NOT commit results.tsv — keep it untracked).
8. If `val_score` improved (higher): keep the commit, the branch advances.
9. If `val_score` is equal or worse: `git reset --hard HEAD~1` to revert.
10. Go to step 1.

## Logging Results

Log every experiment to `results.tsv` (tab-separated). The TSV has a header row and 5 columns:

```
commit	val_score	train_seconds	status	description
```

1. git commit hash (short, 7 chars)
2. val_score achieved (e.g. 0.4231) — use 0.0000 for crashes
3. training time in seconds (e.g. 1823.5) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	val_score	train_seconds	status	description
a1b2c3d	0.3215	0.0	keep	baseline (base model, no training)
b2c3d4e	0.4102	1823.5	keep	lr=2e-5, lora_rank=16
c3d4e5f	0.3980	1756.2	discard	lr=1e-4, lora_rank=16
d4e5f6g	0.0000	0.0	crash	lora_rank=128 (OOM)
```

**Do NOT commit results.tsv** — keep it untracked by git. Only `train.py` changes go into the git history.

## Autonomy

**NEVER STOP.** Once the experiment loop has begun (after initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — re-read train.py for new angles, try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

## Crash Handling

- If a run crashes with a trivial bug (typo, missing import, shape mismatch), fix it and re-run.
- If the idea is fundamentally broken (OOM, conceptually wrong), log it as `crash` in results.tsv, revert, and move on.
- If a run hangs or exceeds the timeout, kill it and treat as a crash.

## Simplicity Criterion

All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win.

When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude:
- A 0.005 val_score improvement that adds 30 lines of hacky code? Probably not worth it.
- A 0.005 val_score improvement from deleting code? Definitely keep.
- Same score but simpler code? Keep.

## What `train.py` Should Explore

- LoRA hyperparameters (rank, alpha, dropout, target modules)
- Learning rate and optimizer choice
- Scheduler type and warmup steps
- Step budget (MAX_STEPS)
- Batch-level configuration (batch size, gradient accumulation)
- Training prompt formatting (USER_PROMPT_PREFIX, ANSWER_STYLE)
- Gradient checkpointing
- Full SFT vs LoRA mode (LoRA recommended on MPS)

## Plotting Progress

After the experiment loop ends (or periodically), generate a progress plot:

```python
import matplotlib.pyplot as plt
import csv

rows = []
with open("results.tsv") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        rows.append(row)

indices = list(range(len(rows)))
scores = [float(r["val_score"]) for r in rows]
best_so_far = []
best = 0
for s in scores:
    best = max(best, s)
    best_so_far.append(best)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(indices, scores, "o-", label="val_score per experiment", alpha=0.6)
ax.plot(indices, best_so_far, "s-", label="best so far", color="green")
ax.set_xlabel("Experiment #")
ax.set_ylabel("val_score (ANLS)")
ax.set_title("Autoresearch Progress")
ax.legend()
ax.grid(True, alpha=0.3)
labels = [r["description"][:20] for r in rows]
ax.set_xticks(indices)
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
plt.tight_layout()
plt.savefig("autoresearch_progress.png", dpi=150)
print("Saved autoresearch_progress.png")
```

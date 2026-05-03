# autoresearch-qwen

Autonomous research loop for improving [Qwen/Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) on the official [HuggingFaceM4/DocumentVQA](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) benchmark.

The repo is designed for agentic training research: the benchmark and evaluator stay fixed, while an agent iterates on `train.py`, runs training, measures the result on the full validation split, and keeps only real gains. The project is inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch), but scoped to a concrete public VLM benchmark with a reproducible contract.

If this project is useful for your research, evals, or agent workflows, please star the repo.

## Branches

This repository now contains the two previously separate codebases as different branches:

| Branch | Target hardware | Status |
| --- | --- | --- |
| `main` | NVIDIA / CUDA multi-GPU | Primary branch. Uses `torchrun`, supports DeepSpeed configs, and is the recommended branch for fast experiment cycles. |
| `mlx` | Apple Silicon / MPS | Historical branch imported from the former `autoresearch-qwen-mlx` repository and preserved here as the Mac-focused variant. |

Use the README on each branch for branch-specific commands. On `main` the entrypoint is `./run_experiment.sh`; on `mlx` it is `uv run python run_experiment.py`.

## Why This Repo Exists

- Fixed benchmark: full official DocVQA `train`, `validation`, and `test` splits
- Fixed evaluator: validation score is always computed by the repository evaluator
- One mutable surface: agents are expected to edit `train.py`
- Reproducible loop: prepare, train, evaluate, keep or discard, repeat
- Public benchmark mindset: improvements should come from better training decisions, not from moving the goalposts

## Benchmark Contract

| Component | Contract |
| --- | --- |
| Base model | `Qwen/Qwen3-VL-4B-Instruct` |
| Dataset | `HuggingFaceM4/DocumentVQA` official splits |
| Training split | Full `train` split |
| Validation split | Full `validation` split |
| Test split | Full blind `test` split |
| Metric | Mean ANLS on the full validation split |
| Mutable file | `train.py` |
| Fixed files | `evaluate.py`, `src/`, benchmark contract, submission tooling |

More benchmark details are documented in [benchmarks/README.md](benchmarks/README.md).

## How The Loop Works

```text
prepare.py          Download dataset + model snapshot
      |
train.py            Mutable training code (the agent edits this)
      |
evaluate.py         Fixed validation evaluator / blind test exporter
      |
run_experiment.sh   One full train -> eval iteration on main
```

The only objective is to maximize `val_score`, defined as mean ANLS on the full official validation split.

## Quick Start (`main`)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run python prepare.py
uv run autoresearch-qwen doctor
uv run python evaluate.py --base-only --split validation
./run_experiment.sh | tee run.log
```

Useful follow-up commands:

```bash
uv run python analysis.py
uv run python submit_test.py
```

## Repository Layout

```text
train.py                            Mutable training code
evaluate.py                         Fixed evaluator
run_experiment.sh                   One-command train -> eval pipeline on main
analysis.py                         Result visualization
prepare.py                          Dataset + model downloader
submit_test.py                      Blind test export + submission packaging
check_submission.py                 Submission validator
program.md                          Full agent protocol
benchmarks/README.md                Benchmark definition
configs/                            DeepSpeed configs for multi-GPU runs
src/autoresearch_qwen/              Fixed library code
```

## Running An Agent

The full experiment protocol lives in [program.md](program.md). A practical starting prompt is:

```text
Read the entire repository, especially README.md and program.md. You may read all files for context, but only edit train.py. Run `uv run autoresearch-qwen doctor --json`, record a `--base-only` validation baseline, then start the autoresearch loop. Parse `artifacts/last_result.json` after each run and keep only changes that improve val_score.
```

## Results, Analysis, and Submission

- `artifacts/last_result.json` stores the latest train/eval result payload
- `analysis.py` plots experiment progress from accumulated results
- `submit_test.py` exports predictions for the blind DocVQA `test` split
- `check_submission.py` validates a submission bundle locally before upload

## Contributing

Issues and pull requests are welcome, especially for:

- stronger training recipes that respect the benchmark contract
- better experiment tooling and reproducibility
- clearer docs and onboarding
- hardware-specific improvements that belong on a dedicated branch

If you want to change the benchmark contract itself, open an issue first so the rationale is explicit.

## Acknowledgements

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch) for the original autonomous research-loop framing
- [Qwen](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) for the base vision-language model
- [Hugging Face M4](https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA) for the public DocVQA dataset release

## Star Trend

[![Star History Chart](https://api.star-history.com/svg?repos=wadeKeith/autoresearch-qwen&type=Date&cache=2026050308)](https://star-history.com/#wadeKeith/autoresearch-qwen&Date)

## License

[MIT](LICENSE)

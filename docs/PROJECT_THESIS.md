# Project Thesis

This repository should behave like `autoresearch-mlx`, but for `Qwen3-VL-4B`.

That means the improvement surface must be training code, not a tiny prompt file.

The smallest workable version is:

- fixed public `DocVQA` official split contract
- fixed ANLS evaluation on `validation`
- full SFT with optional LoRA using `transformers.Trainer`
- a single mutable `train.py`
- a single scalar metric

This is much closer to the real spirit of autoresearch than a generic VLM demo.

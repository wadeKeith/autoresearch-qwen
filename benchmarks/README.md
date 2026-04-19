# Benchmark

The benchmark uses the full official DocVQA splits, not a small subset.

- source dataset: `HuggingFaceM4/DocumentVQA`
- train split: **all** official `train` examples (~39K)
- validation split: **all** official `validation` examples (~5K)
- test split: **all** official `test` examples (blind, no answers)
- task: answer a question by reading a single document image
- score: mean ANLS over full validation answers
- model: `Qwen/Qwen3-VL-4B-Instruct`

The full official splits are used for both training and evaluation.
Experiment speed depends on hardware (8x A100-80G recommended) and hyperparameters;
a typical train+eval iteration takes 30-90 minutes.

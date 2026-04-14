# Benchmark Plan

The benchmark is intentionally small, public, and fixed.

It is designed for repeated `Qwen3-VL-4B` training experiments:

- source dataset: `HuggingFaceM4/DocumentVQA`
- fixed question types: `table/list`, `free_text`, `figure/diagram`
- train split: 32 fixed examples from `train`
- validation split: 8 fixed examples from `validation`
- task: answer a question by reading a single document image
- score: mean ANLS over validation answers

This setup is small enough for iterative autoresearch and large enough for visible score movement.

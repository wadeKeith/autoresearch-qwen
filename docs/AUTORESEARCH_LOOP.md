# Autoresearch Loop

## Fixed

- benchmark downloader
- official `train` / `validation` / `test` split contract
- base model
- evaluator
- score definition

## Mutable

- `train.py`

## Objective

Improve `val_score` by changing the training code while leaving evaluation fixed.

## Good Search Space

- full-SFT hyperparameters
- optimizer choice
- gradient checkpointing
- prompt packing during SFT
- target module selection when LoRA is enabled
- scheduler and epoch budget

## Bad Search Space

- editing the benchmark
- editing the evaluator
- changing the base model away from `Qwen3-VL-4B`

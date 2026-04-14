"""Central project metadata exposed by the CLI."""

PROJECT_NAME = "autoresearch-qwen-mlx"
TAGLINE = "Training-code autoresearch for Qwen3-VL-4B on the official DocVQA splits."

THESIS_LINES = [
    "Use a fixed benchmark and fixed evaluator around a mutable train.py.",
    "Improve Qwen3-VL-4B with real transformers Trainer training on the official DocVQA train split.",
    "Use validation for visible ANLS and keep test as a blind prediction split.",
    "Optimize for repeated public-benchmark experiments.",
]

LOOP_LINES = [
    "prepare.py snapshot-downloads the full official DocumentVQA dataset.",
    "train.py learns on train with LoRA (default) or full SFT.",
    "evaluate.py runs the final visible validation score after training.",
    "evaluate.py exports blind test predictions when needed.",
    "run_experiment.py is the one-command train plus eval loop.",
]

FILES_LINES = [
    "program.md: autonomous experiment instructions.",
    "prepare.py: official-snapshot downloader.",
    "train.py: mutable Trainer training code with LoRA default and optional full SFT.",
    "evaluate.py: fixed validation scorer and test prediction exporter.",
    "submit_test.py: one-command test export, validation, and packaging.",
    "run_experiment.py: isolated experiment runner.",
    "results.tsv: keep and discard history (untracked by git).",
]

BENCHMARK_LINES = [
    "Use the official HuggingFaceM4/DocumentVQA train, validation, and test splits directly.",
    "Train on train, evaluate after training on the full validation split.",
    "Keep test as a blind split because its answers are hidden in the public dataset.",
    "Use LoRA on Qwen3-VL-4B by default, with full SFT still available as an option.",
    "Report one primary metric: val_score.",
]

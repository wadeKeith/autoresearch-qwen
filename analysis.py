"""Visualize autoresearch experiment progress from results.tsv.

Usage:
    uv run python analysis.py                   # default: autoresearch_progress.png
    uv run python analysis.py -o my_plot.png    # custom output path
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS_PATH = ROOT / "results.tsv"


def load_results(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        print(f"ERROR: {path} not found.", file=sys.stderr)
        sys.exit(1)
    with open(path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot autoresearch experiment progress.")
    parser.add_argument("-i", "--input", type=Path, default=RESULTS_PATH)
    parser.add_argument("-o", "--output", type=Path, default=ROOT / "autoresearch_progress.png")
    args = parser.parse_args()

    rows = load_results(args.input)
    if not rows:
        print("No experiment rows found in results.tsv.", file=sys.stderr)
        sys.exit(1)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required. Run `uv sync` to install project dependencies.", file=sys.stderr)
        sys.exit(1)

    indices = list(range(len(rows)))
    scores = [float(r["val_score"]) for r in rows]
    best_so_far = []
    best = 0.0
    for s in scores:
        best = max(best, s)
        best_so_far.append(best)

    # Memory subplot (if column exists)
    has_memory = "memory_gb" in rows[0]

    nplots = 2 if has_memory else 1
    fig, axes = plt.subplots(nplots, 1, figsize=(12, 5 * nplots), squeeze=False)

    # --- Score plot ---
    ax = axes[0, 0]
    colors = {"keep": "green", "discard": "orange", "crash": "red"}
    for i, row in enumerate(rows):
        c = colors.get(row.get("status", ""), "gray")
        ax.scatter(i, scores[i], color=c, zorder=3, s=40)
    ax.plot(indices, scores, "o-", alpha=0.3, color="steelblue", markersize=3, label="val_score")
    ax.plot(indices, best_so_far, "s-", color="green", markersize=4, label="best so far")
    ax.set_xlabel("Experiment #")
    ax.set_ylabel("val_score (ANLS)")
    ax.set_title("Autoresearch Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)
    labels = [r.get("description", "")[:25] for r in rows]
    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

    # --- Memory plot ---
    if has_memory:
        ax2 = axes[1, 0]
        mem = [float(r["memory_gb"]) for r in rows]
        ax2.bar(indices, mem, color="steelblue", alpha=0.7)
        ax2.set_xlabel("Experiment #")
        ax2.set_ylabel("Peak VRAM (GB)")
        ax2.set_title("Memory Usage per Experiment")
        ax2.set_xticks(indices)
        ax2.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

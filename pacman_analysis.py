"""Analyze Ms. Pac-Man AI run data and produce article-ready charts.

Reads all data from output/runs/ and output/eval/, aggregates by model,
and generates matplotlib charts.

Usage:
    python pacman_analysis.py                  # generate all charts
    python pacman_analysis.py --out figures/   # custom output directory
    python pacman_analysis.py --json           # also dump aggregated data as JSON

Outputs PNGs to output/analysis/ (or --out).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS_DIR = SCRIPT_DIR / "output" / "runs"
EVAL_DIR = SCRIPT_DIR / "output" / "eval"
DEFAULT_OUT = SCRIPT_DIR / "output" / "analysis"

MODEL_SHORT_NAMES = {
    "mlx-community/Qwen3-VL-4B-Instruct-4bit": "Qwen3-VL 4B",
    "mlx-community/Qwen3-VL-4B-Instruct-4bit (schematic)": "Qwen3-VL 4B",
    "mlx-community/Qwen3-VL-4B-Instruct-4bit (raw)": "Qwen3-VL 4B (raw)",
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit": "Qwen2.5-VL 3B",
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit (schematic)": "Qwen2.5-VL 3B",
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit (raw)": "Qwen2.5-VL 3B (raw)",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit": "Qwen2.5-VL 7B",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit (schematic)": "Qwen2.5-VL 7B",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit (raw)": "Qwen2.5-VL 7B (raw)",
    "mlx-community/SmolVLM-Instruct-4bit": "SmolVLM",
    "mlx-community/SmolVLM-Instruct-4bit (schematic)": "SmolVLM",
    "mlx-community/SmolVLM-Instruct-4bit (raw)": "SmolVLM (raw)",
    "mlx-community/pixtral-12b-2409-4bit": "Pixtral 12B",
    "mlx-community/pixtral-12b-4bit": "Pixtral 12B",
    "mlx-community/pixtral-12b-4bit (schematic)": "Pixtral 12B",
    "mlx-community/pixtral-12b-4bit (raw)": "Pixtral 12B (raw)",
    "mlx-community/SmolVLM-Instruct-4bit": "SmolVLM",
    "mlx-community/gemma-3-4b-it-4bit": "Gemma 3 4B",
    "mlx-community/gemma-3-4b-it-4bit (schematic)": "Gemma 3 4B",
    "mlx-community/gemma-3-4b-it-4bit (raw)": "Gemma 3 4B (raw)",
    "vikhyatk/moondream2": "Moondream2",
    "vikhyatk/moondream2 (schematic)": "Moondream2",
    "vikhyatk/moondream2 (raw)": "Moondream2 (raw)",
    "mlx-community/moondream2-bf16": "Moondream2 bf16",
    "mlx-community/paligemma2-3b-pt-448-4bit": "PaliGemma2 3B",
    "mlx-community/llava-v1.6-mistral-7b-4bit": "LLaVA 1.6 7B",
    "claude-opus-4-0-20250115": "Claude Opus 4",
    "random baseline": "Random",
}

PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
    "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7",
    "#9C755F", "#BAB0AC",
]


def short_name(model: str) -> str:
    if model in MODEL_SHORT_NAMES:
        return MODEL_SHORT_NAMES[model]
    return model.split("/")[-1] if "/" in model else model


def load_all_runs() -> list[dict]:
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for d in sorted(RUNS_DIR.iterdir()):
        sf = d / "summary.json"
        if sf.exists():
            try:
                s = json.loads(sf.read_text())
                s["_dir"] = d.name
                s["_run_dir"] = str(d)
                runs.append(s)
            except (json.JSONDecodeError, OSError):
                pass
    return runs


def load_turns(run_dir: str | Path) -> list[dict]:
    tf = Path(run_dir) / "turns.jsonl"
    if not tf.exists():
        return []
    entries = []
    for line in tf.read_text().strip().split("\n"):
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries


def load_eval_summaries() -> list[dict]:
    evals = []
    if not EVAL_DIR.exists():
        return evals
    for d in sorted(EVAL_DIR.iterdir()):
        sf = d / "eval_summary.json"
        if sf.exists():
            try:
                evals.append(json.loads(sf.read_text()))
            except (json.JSONDecodeError, OSError):
                pass
    return evals


def aggregate_by_model(runs: list[dict]) -> dict[str, list[dict]]:
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        player = r.get("player", "unknown")
        by_model[player].append(r)
    return dict(by_model)


def style_setup():
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor": "#16213e",
        "axes.edgecolor": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "text.color": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "grid.color": "#2a3a5e",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 12,
        "figure.dpi": 150,
    })


def chart_score_comparison(by_model: dict, out_dir: Path):
    """Bar chart: average score per model with error bars."""
    models = []
    avgs = []
    stds = []
    counts = []
    for player, games in sorted(by_model.items(), key=lambda x: -mean(g.get("final_score") or 0 for g in x[1])):
        scores = [g.get("final_score") or 0 for g in games]
        models.append(short_name(player))
        avgs.append(mean(scores))
        stds.append(stdev(scores) if len(scores) > 1 else 0)
        counts.append(len(scores))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 6))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]
    bars = ax.bar(models, avgs, yerr=stds, capsize=5, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, avg, n in zip(bars, avgs, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{avg:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 5,
                f"n={n}", ha="center", va="bottom", fontsize=8, alpha=0.7)

    ax.set_ylabel("Score")
    ax.set_title("Ms. Pac-Man: Average Score by Model", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, max(avgs) * 1.3 if avgs else 100)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "score_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  score_comparison.png")


def chart_score_distribution(by_model: dict, out_dir: Path):
    """Box plot showing score distribution per model."""
    data = []
    labels = []
    for player, games in sorted(by_model.items(), key=lambda x: -mean(g.get("final_score") or 0 for g in x[1])):
        scores = [g.get("final_score") or 0 for g in games]
        data.append(scores)
        labels.append(short_name(player))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 6))
    bp = ax.boxplot(data, patch_artist=True, tick_labels=labels, widths=0.6,
                    medianprops=dict(color="white", linewidth=2))

    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(PALETTE[i % len(PALETTE)])
        patch.set_alpha(0.8)

    for i, scores in enumerate(data):
        jitter = np.random.normal(0, 0.04, len(scores))
        ax.scatter([i + 1 + j for j in jitter], scores,
                   color="white", alpha=0.5, s=20, zorder=3)

    ax.set_ylabel("Score")
    ax.set_title("Score Distribution by Model", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "score_distribution.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  score_distribution.png")


def chart_turn_progression(by_model: dict, out_dir: Path):
    """Line chart: score over turns for each model (best game per model)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (player, games) in enumerate(sorted(by_model.items())):
        best_game = max(games, key=lambda g: g.get("final_score") or 0)
        run_dir = best_game.get("_run_dir")
        if not run_dir:
            continue
        turns = load_turns(run_dir)
        if not turns:
            continue

        turn_nums = []
        scores = []
        for t in turns:
            if "turn" in t and "score" in t and t["score"] is not None:
                turn_nums.append(t["turn"])
                scores.append(t["score"])

        if turn_nums:
            label = f"{short_name(player)} (best: {best_game.get('final_score', 0)})"
            ax.plot(turn_nums, scores, marker="o", markersize=3,
                    color=PALETTE[i % len(PALETTE)], linewidth=2, label=label, alpha=0.9)

    ax.set_xlabel("Turn")
    ax.set_ylabel("Score")
    ax.set_title("Score Progression (Best Game per Model)", fontsize=16, fontweight="bold", pad=15)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(linestyle="--")
    plt.tight_layout()
    fig.savefig(out_dir / "turn_progression.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  turn_progression.png")


def chart_latency(by_model: dict, out_dir: Path):
    """Bar chart: average LLM inference time per model."""
    models = []
    latencies = []

    for player, games in sorted(by_model.items()):
        all_lat = []
        for g in games:
            run_dir = g.get("_run_dir")
            if not run_dir:
                continue
            for t in load_turns(run_dir):
                lat = t.get("llm_seconds") or t.get("think_seconds")
                if lat and lat > 0:
                    all_lat.append(lat)
        if all_lat:
            models.append(short_name(player))
            latencies.append(mean(all_lat))

    if not models:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]
    bars = ax.bar(models, latencies, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, lat in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{lat:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Seconds per Move")
    ax.set_title("Average Inference Latency per Move", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "latency.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  latency.png")


def chart_decision_quality(by_model: dict, out_dir: Path):
    """Stacked bar: % of moves that scored vs. didn't, by model."""
    models = []
    scored_pcts = []
    neutral_pcts = []
    total_moves = []

    for player, games in sorted(by_model.items(), key=lambda x: -mean(g.get("final_score") or 0 for g in x[1])):
        scored = 0
        total = 0
        for g in games:
            run_dir = g.get("_run_dir")
            if not run_dir:
                continue
            turns = load_turns(run_dir)
            prev_score = None
            for t in turns:
                s = t.get("score")
                if s is not None and prev_score is not None:
                    total += 1
                    if s > prev_score:
                        scored += 1
                prev_score = s

        if total > 0:
            models.append(short_name(player))
            scored_pcts.append(scored / total * 100)
            neutral_pcts.append((total - scored) / total * 100)
            total_moves.append(total)

    if not models:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    x = range(len(models))
    ax.bar(x, scored_pcts, color="#59A14F", label="Scored points", alpha=0.9, edgecolor="white", linewidth=0.5)
    ax.bar(x, neutral_pcts, bottom=scored_pcts, color="#E15759", label="No score gain", alpha=0.7, edgecolor="white", linewidth=0.5)

    for i, (sp, n) in enumerate(zip(scored_pcts, total_moves)):
        ax.text(i, sp / 2, f"{sp:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(i, 102, f"n={n}", ha="center", va="bottom", fontsize=8, alpha=0.7)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=25, ha="right")
    ax.set_ylabel("% of Moves")
    ax.set_title("Decision Quality: % of Moves That Scored Points", fontsize=16, fontweight="bold", pad=15)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--")
    plt.tight_layout()
    fig.savefig(out_dir / "decision_quality.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  decision_quality.png")


def chart_moves_survived(by_model: dict, out_dir: Path):
    """Bar chart: average moves survived before losing a life."""
    models = []
    avg_moves = []
    for player, games in sorted(by_model.items(), key=lambda x: -mean(g.get("moves_played", 0) for g in x[1])):
        m = [g.get("moves_played", 0) for g in games]
        models.append(short_name(player))
        avg_moves.append(mean(m))

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.5), 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(models))]
    bars = ax.bar(models, avg_moves, color=colors, edgecolor="white", linewidth=0.5, alpha=0.9)

    for bar, m in zip(bars, avg_moves):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{m:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_ylabel("Moves")
    ax.set_title("Average Moves Survived Before Life Lost", fontsize=16, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "moves_survived.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  moves_survived.png")


def dump_summary_json(by_model: dict, out_dir: Path):
    """Write aggregated stats as JSON for the dashboard artifact."""
    summary = {}
    for player, games in by_model.items():
        scores = [g.get("final_score") or 0 for g in games]
        moves = [g.get("moves_played", 0) for g in games]

        all_lat = []
        scored_moves = 0
        total_counted = 0
        for g in games:
            run_dir = g.get("_run_dir")
            if not run_dir:
                continue
            turns = load_turns(run_dir)
            prev_score = None
            for t in turns:
                lat = t.get("llm_seconds") or t.get("think_seconds")
                if lat and lat > 0:
                    all_lat.append(lat)
                s = t.get("score")
                if s is not None and prev_score is not None:
                    total_counted += 1
                    if s > prev_score:
                        scored_moves += 1
                prev_score = s

        summary[short_name(player)] = {
            "full_model": player,
            "games": len(games),
            "avg_score": round(mean(scores), 1),
            "median_score": round(median(scores), 1),
            "best_score": max(scores),
            "worst_score": min(scores),
            "std_score": round(stdev(scores), 1) if len(scores) > 1 else 0,
            "avg_moves": round(mean(moves), 1),
            "avg_latency": round(mean(all_lat), 2) if all_lat else None,
            "scoring_move_pct": round(scored_moves / total_counted * 100, 1) if total_counted else None,
            "individual_scores": scores,
        }

    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  summary.json")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Ms. Pac-Man AI Analysis & Charts")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory for charts")
    parser.add_argument("--json", action="store_true", help="Also dump aggregated JSON")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading run data...")
    runs = load_all_runs()
    if not runs:
        sys.exit("No runs found in output/runs/")

    by_model = aggregate_by_model(runs)
    print(f"Found {len(runs)} runs across {len(by_model)} model(s):")
    for player, games in by_model.items():
        scores = [g.get("final_score") or 0 for g in games]
        print(f"  {short_name(player):25s}  {len(games):3d} games  avg {mean(scores):.0f}  best {max(scores)}")

    print(f"\nGenerating charts in {out_dir}/")
    style_setup()

    chart_score_comparison(by_model, out_dir)
    chart_score_distribution(by_model, out_dir)
    chart_turn_progression(by_model, out_dir)
    chart_latency(by_model, out_dir)
    chart_decision_quality(by_model, out_dir)
    chart_moves_survived(by_model, out_dir)

    if args.json:
        dump_summary_json(by_model, out_dir)

    print(f"\nDone! {len(list(out_dir.glob('*.png')))} charts generated.")


if __name__ == "__main__":
    main()

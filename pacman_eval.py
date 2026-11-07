"""Run multiple Ms. Pac-Man games and produce an evaluation summary.

Usage:
    python pacman_eval.py --runs 5
    python pacman_eval.py --runs 10 --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    python pacman_eval.py --runs 5 --turns 40 --keep-playing
    python pacman_eval.py --runs 3 --baseline random

Results are saved to output/eval/<timestamp>/eval_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median

SCRIPT_DIR = Path(__file__).resolve().parent
BRAIN_SCRIPT = SCRIPT_DIR / "pacman_brain.py"
RUNS_DIR = SCRIPT_DIR / "output" / "runs"
EVAL_DIR = SCRIPT_DIR / "output" / "eval"

sys.path.insert(0, str(SCRIPT_DIR))
from pacman_brain import DEFAULT_MODEL, KNOWN_MODELS


def find_latest_run(before: float) -> Path | None:
    """Find the most recent run directory created after `before` timestamp."""
    if not RUNS_DIR.exists():
        return None
    best = None
    best_time = 0.0
    for d in RUNS_DIR.iterdir():
        if not d.is_dir():
            continue
        summary = d / "summary.json"
        if summary.exists():
            t = summary.stat().st_mtime
            if t > before and t > best_time:
                best = d
                best_time = t
    return best


def run_one_game(args: argparse.Namespace, game_num: int, total: int) -> dict | None:
    """Run a single game via pacman_brain.py and return its summary."""
    print(f"\n{'='*60}")
    print(f"  GAME {game_num}/{total}")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, str(BRAIN_SCRIPT),
        "--model", args.model,
        "--mesen", args.mesen,
        "--rom", args.rom,
        "--save", args.save,
        "--lua", args.lua,
    ]
    if args.turns:
        cmd += ["--turns", str(args.turns)]
    if args.keep_playing:
        cmd += ["--keep-playing"]
    if args.baseline:
        cmd += ["--baseline", args.baseline]
    if args.raw:
        cmd += ["--raw"]

    before = time.time()
    try:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  Brain exited with code {result.returncode}")
        subprocess.run(["osascript", "-e", 'quit app "Mesen"'], capture_output=True)
    except KeyboardInterrupt:
        print("\n  Interrupted — stopping evaluation.")
        subprocess.run(["pkill", "-f", "Mesen"], capture_output=True)
        return None

    time.sleep(5)

    run_dir = find_latest_run(before)
    if not run_dir:
        print(f"  Could not find results for game {game_num}")
        print(f"  Runs dir: {RUNS_DIR}")
        if RUNS_DIR.exists():
            dirs = sorted(RUNS_DIR.iterdir())
            print(f"  Available runs: {[d.name for d in dirs[-5:]]}")
        return None

    summary_path = run_dir / "summary.json"
    try:
        data = json.loads(summary_path.read_text())
        data["run_dir"] = str(run_dir.name)
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Failed to read summary: {e}")
        return None


def print_report(results: list[dict], model: str, eval_dir: Path):
    """Print and save the evaluation report."""
    scores = [r["final_score"] or 0 for r in results]
    turns = [r["moves_played"] for r in results]
    lives_lost = [r.get("lives_lost", 0) for r in results]

    report_lines = []

    def out(line=""):
        report_lines.append(line)
        print(line)

    out()
    out("=" * 60)
    out("  MS. PAC-MAN EVALUATION REPORT")
    out("=" * 60)
    out(f"  Model:    {model}")
    out(f"  Games:    {len(results)}")
    out(f"  Date:     {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    out()

    out("  GAME RESULTS")
    out("  " + "-" * 50)
    out(f"  {'#':>3}  {'Score':>7}  {'Turns':>6}  {'Lives':>6}  {'End Reason'}")
    out("  " + "-" * 50)
    for i, r in enumerate(results, 1):
        s = r["final_score"] or 0
        t = r["moves_played"]
        ll = r.get("lives_lost", 0)
        reason = r.get("end_reason", "?")
        out(f"  {i:>3}  {s:>7}  {t:>6}  {ll:>6}  {reason}")
    out("  " + "-" * 50)
    out()

    out("  STATS")
    out("  " + "-" * 50)
    out(f"  Best score:     {max(scores):>7}")
    out(f"  Worst score:    {min(scores):>7}")
    out(f"  Average score:  {mean(scores):>7.0f}")
    out(f"  Median score:   {median(scores):>7.0f}")
    out(f"  Avg turns:      {mean(turns):>7.1f}")
    out(f"  Avg lives lost: {mean(lives_lost):>7.1f}")
    out("=" * 60)

    eval_summary = {
        "model": model,
        "games": len(results),
        "date": datetime.now().isoformat(timespec="seconds"),
        "best_score": max(scores),
        "worst_score": min(scores),
        "avg_score": round(mean(scores), 1),
        "median_score": round(median(scores), 1),
        "avg_turns": round(mean(turns), 1),
        "avg_lives_lost": round(mean(lives_lost), 1),
        "results": results,
    }
    eval_dir.mkdir(parents=True, exist_ok=True)
    summary_path = eval_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(eval_summary, indent=2), encoding="utf-8")
    report_path = eval_dir / "eval_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    out(f"\n  Saved: {summary_path}")
    out(f"  Saved: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Ms. Pac-Man Multi-Run Evaluation")
    parser.add_argument("--runs", type=int, default=5, help="Number of games to play")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID. Known: " + ", ".join(
            m.split("/")[-1] for m in KNOWN_MODELS
        ),
    )
    parser.add_argument("--turns", type=int, default=0, help="Max turns per game (0=unlimited)")
    parser.add_argument("--keep-playing", action="store_true",
                        help="Don't stop on first life lost")
    parser.add_argument("--label", default=None, help="Model name for the report (display only)")
    parser.add_argument("--baseline", choices=("llm", "random"), default=None)
    parser.add_argument("--raw", action="store_true",
                        help="Send raw screenshots (skip OpenCV preprocessing)")
    parser.add_argument("--mesen", default=str(
        Path(__file__).resolve().parent.parent.parent.parent / "Mesen.app"))
    parser.add_argument("--rom", default=str(
        Path(__file__).resolve().parent.parent.parent.parent / "rom" / "Ms. Pac-Man (USA).sfc"))
    parser.add_argument("--save", default=str(
        Path.home() / "Library" / "Application Support" / "MesenCE"
        / "SaveStates" / "Ms. Pac-Man (USA)_1.mss"))
    parser.add_argument("--lua", default=str(
        Path(__file__).resolve().parent / "pacman_director.lua"))
    args = parser.parse_args()

    label = args.label or args.baseline or args.model
    print(f"Ms. Pac-Man Evaluation: {args.runs} games with {label}")
    print(f"Max {args.turns} turns per game (0=unlimited)")

    eval_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_out = EVAL_DIR / eval_ts

    results = []
    for i in range(1, args.runs + 1):
        data = run_one_game(args, i, args.runs)
        if data is None and i > 1:
            break
        if data:
            results.append(data)

    if results:
        print_report(results, label, eval_out)
    else:
        print("No results collected.")


if __name__ == "__main__":
    main()

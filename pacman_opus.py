"""Interactive Ms. Pac-Man player for Claude Opus (via Claude Code).

Handles game loop bookkeeping (Mesen launch, screenshots, WRAM tracking,
logging) while an external player provides moves via stdin.

Usage from Claude Code:
    1. Run: python pacman_opus.py
    2. When prompted, read the screenshot at output/screen.png
    3. Type a move: up/down/left/right [frames]  (frames defaults to 60)
    4. Repeat until done. Type 'quit' to stop.

    python pacman_opus.py --turns 30          # limit to 30 turns
    python pacman_opus.py --keep-playing      # don't stop on life lost
    python pacman_opus.py --player "Claude Opus 4"  # custom label
    python pacman_opus.py --no-launch         # don't launch Mesen (already running)
    python pacman_opus.py --preprocess        # also save schematic view

Logs to output/runs/<timestamp>/ in the same format as pacman_brain.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent

DEFAULT_MESEN = REPO_ROOT / "Mesen.app"
DEFAULT_ROM = REPO_ROOT / "rom" / "Ms. Pac-Man (USA).sfc"
DEFAULT_SAVE = (
    Path.home()
    / "Library"
    / "Application Support"
    / "MesenCE"
    / "SaveStates"
    / "Ms. Pac-Man (USA)_1.mss"
)
DEFAULT_LUA = SCRIPT_DIR / "pacman_director.lua"

MESEN_DATA = Path.home() / "Library" / "Application Support" / "MesenCE"
SCRIPT_DATA = MESEN_DATA / "LuaScriptData" / "pacman_director"
CMD_FILE = SCRIPT_DATA / "command.txt"
STATUS_FILE = SCRIPT_DATA / "output" / "status.txt"
SCREEN_FILE = SCRIPT_DATA / "output" / "screen.png"
WRAM_FILE = SCRIPT_DATA / "output" / "wram.bin"
LOCAL_OUTPUT = SCRIPT_DIR / "output"
RUNS_DIR = LOCAL_OUTPUT / "runs"

VALID_DIRECTIONS = ("up", "down", "left", "right")

SCHEMATIC_PROMPT = """\
This is a simplified map of the Ms. Pac-Man game.

Legend:
- Green circle labeled YOU = Pac-Man (you control her)
- White dots = pellets (eat them for points)
- Red circle labeled G = ghost (avoid — they kill you)
- Green arrows with numbers = open directions (number = pellet count that way)
- Red X = wall (blocked, cannot move there)

Your task:
1. Look at the arrows around YOU. Green arrows show open directions; red X means blocked.
2. The number on each green arrow is how many pellets are in that direction — prefer higher numbers.
3. If a ghost (red G) is nearby in a direction, avoid that direction.
4. Pick the open direction with the most pellets and no nearby ghost.

Answer in this format:
OPEN: <list of open directions with pellet counts>
BLOCKED: <list of blocked directions>
DANGER: <direction of nearest ghost, or none>
MOVE: <your chosen direction>
"""

RAW_PROMPT = """\
This is a Pac-Man game. Yellow=Pac-Man, magenta=walls, white dots=pellets.
Ghosts: red, pink, orange, cyan blobs with eyes — avoid them.
Exception: blue/white ghosts are EDIBLE — chase them for bonus points!
Answer briefly:
PACMAN: <position>
GHOSTS: <color and position of each ghost>
PELLETS: <which directions have pellets>
DANGER: <which directions have normal ghosts>
GO: <one non-diagonal direction>
"""

LIVES_ADDR = 0x028C
SCORE_ADDR = 0x02A2
SCORE_BYTES = 3


def decode_bcd(snap: bytes, addr: int, width: int) -> int | None:
    raw = snap[addr : addr + width]
    if len(raw) < width:
        return None
    digits = "".join(f"{b:02x}" for b in reversed(raw))
    return int(digits) if digits.isdigit() else None


class GameTracker:
    def __init__(self) -> None:
        self.lives: int | None = None
        self.start_lives: int | None = None
        self.lives_lost = 0
        self.score: int | None = None

    def update(self, snap: bytes) -> list[str]:
        events: list[str] = []
        lives = snap[LIVES_ADDR]
        if self.start_lives is None:
            self.start_lives = lives
        if self.lives is not None:
            if lives < self.lives:
                self.lives_lost += self.lives - lives
                events.append(f"LIFE LOST ({self.lives} -> {lives})")
            elif lives > self.lives:
                events.append(f"EXTRA LIFE ({self.lives} -> {lives})")
        self.lives = lives

        value = decode_bcd(snap, SCORE_ADDR, SCORE_BYTES)
        if value is not None:
            if self.score is not None and value != self.score:
                events.append(f"score {self.score} -> {value} ({value - self.score:+d})")
            self.score = value
        return events


def read_status() -> str:
    if STATUS_FILE.exists():
        return STATUS_FILE.read_text().strip()
    return "unknown"


def screen_mtime() -> float:
    try:
        return SCREEN_FILE.stat().st_mtime
    except OSError:
        return 0.0


def send_move(direction: str, frames: int) -> None:
    CMD_FILE.write_text(f"{direction} {frames}\n")


def wait_for_status(target: str, timeout: float = 120) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if read_status() == target:
            return True
        time.sleep(0.5)
    return False


def wait_for_turn_done(prev_mtime: float, timeout: float = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if screen_mtime() != prev_mtime and read_status() == "waiting":
            return True
        time.sleep(0.05)
    return False


def grab_screenshot() -> Path | None:
    if SCREEN_FILE.exists():
        LOCAL_OUTPUT.mkdir(parents=True, exist_ok=True)
        dest = LOCAL_OUTPUT / "screen.png"
        shutil.copy2(SCREEN_FILE, dest)
        return dest
    return None


def read_wram() -> bytes | None:
    try:
        return WRAM_FILE.read_bytes()
    except OSError:
        return None


def clean_output() -> None:
    for folder in (SCRIPT_DATA / "output", LOCAL_OUTPUT):
        if folder.exists():
            for png in folder.glob("*.png"):
                png.unlink(missing_ok=True)
    STATUS_FILE.unlink(missing_ok=True)
    CMD_FILE.unlink(missing_ok=True)
    WRAM_FILE.unlink(missing_ok=True)


def save_schematic(screen_path: Path) -> Path | None:
    try:
        import cv2
        from pacman_vision import analyze_screenshot
        _, _, schematic = analyze_screenshot(str(screen_path))
        out = screen_path.parent / "screen_schematic.png"
        cv2.imwrite(str(out), schematic)
        return out
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Interactive Ms. Pac-Man (for Opus via Claude Code)")
    parser.add_argument("--mesen", default=str(DEFAULT_MESEN))
    parser.add_argument("--rom", default=str(DEFAULT_ROM))
    parser.add_argument("--save", default=str(DEFAULT_SAVE))
    parser.add_argument("--lua", default=str(DEFAULT_LUA))
    parser.add_argument("--turns", type=int, default=0, help="Max turns (0=unlimited)")
    parser.add_argument("--keep-playing", action="store_true")
    parser.add_argument("--player", default="claude-opus-4-0-20250115", help="Player label for logs")
    parser.add_argument("--no-launch", action="store_true", help="Skip launching Mesen (already running)")
    parser.add_argument("--preprocess", action="store_true", help="Also save schematic view")
    parser.add_argument("--raw", action="store_true", help="Use raw screenshot prompt (skip schematic)")
    args = parser.parse_args()

    if not args.no_launch:
        mesen_app = Path(args.mesen)
        rom = Path(args.rom)
        save = Path(args.save)
        lua = Path(args.lua)
        for p, name in [(mesen_app, "Mesen"), (rom, "ROM"), (save, "Save state"), (lua, "Lua script")]:
            if not p.exists():
                sys.exit(f"{name} not found: {p}")

        SCRIPT_DATA.mkdir(parents=True, exist_ok=True)
        (SCRIPT_DATA / "output").mkdir(parents=True, exist_ok=True)
        clean_output()

        print("Launching Mesen...")
        subprocess.Popen(["open", str(mesen_app), "--args", str(rom), str(save), str(lua)])

        print("Waiting for game to load...")
        if not wait_for_status("waiting", timeout=120):
            print("ERROR: Game did not reach waiting state.")
            subprocess.run(["pkill", "-f", "Mesen"], capture_output=True)
            return
    else:
        status = read_status()
        if status != "waiting":
            print(f"Game status: {status} (expected 'waiting')")
            print("Launch Mesen first or remove --no-launch")
            return

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "turns.jsonl"

    tracker = GameTracker()
    print(f"\nPlayer: {args.player}")
    print(f"Run log: {log_path}")
    print("Game ready! Waiting for your moves.\n")
    use_schematic = args.preprocess and not args.raw
    game_prompt = SCHEMATIC_PROMPT if use_schematic else RAW_PROMPT
    print("Commands: up/down/left/right [frames]  |  quit")
    print(f"Screenshot at: {LOCAL_OUTPUT / 'screen.png'}")
    if use_schematic:
        print(f"Schematic at:  {LOCAL_OUTPUT / 'screen_schematic.png'}")
    print(f"Prompt mode:   {'schematic' if use_schematic else 'raw'}")
    print("-" * 50)

    turn = 0
    moves: list[dict] = []
    end_reason = "unknown"

    try:
        with open(log_path, "w", encoding="utf-8") as log:
            while True:
                turn += 1
                if args.turns and turn > args.turns:
                    print(f"\nReached {args.turns} turn limit.")
                    end_reason = "turn_limit"
                    break

                screen = grab_screenshot()
                if not screen:
                    print("  No screenshot, waiting...")
                    time.sleep(2)
                    turn -= 1
                    continue

                if args.preprocess:
                    save_schematic(screen)

                snap = read_wram()
                if snap:
                    for event in tracker.update(snap):
                        print(f"  *** {event}")
                    print(f"  {tracker.lives} lives | {tracker.lives_lost} lost | score {tracker.score}")
                    if tracker.lives_lost > 0 and not args.keep_playing:
                        print("  Life lost — stopping run.")
                        log.write(json.dumps({
                            "event": "life_lost",
                            "time": datetime.now().isoformat(timespec="seconds"),
                            "lives": tracker.lives,
                            "lives_lost": tracker.lives_lost,
                            "score": tracker.score,
                        }) + "\n")
                        log.flush()
                        end_reason = "life_lost"
                        break

                shutil.copy2(screen, run_dir / f"turn_{turn:03d}.png")

                print(f"\n[Turn {turn}] screenshot saved — read it then answer:")
                print(f"--- PROMPT ---\n{game_prompt}--- END PROMPT ---")
                t_start = time.time()

                try:
                    raw = input("> ").strip().lower()
                except EOFError:
                    end_reason = "eof"
                    break

                think_secs = time.time() - t_start

                if raw == "quit":
                    end_reason = "user_quit"
                    break

                parts = raw.split()
                direction = parts[0] if parts else ""
                if direction not in VALID_DIRECTIONS:
                    print(f"  Invalid direction '{direction}'. Use: up/down/left/right")
                    turn -= 1
                    continue
                frames = int(parts[1]) if len(parts) > 1 else 60

                log.write(json.dumps({
                    "turn": turn,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "think_seconds": round(think_secs, 2),
                    "direction": direction,
                    "frames": frames,
                    "lives": tracker.lives,
                    "lives_lost": tracker.lives_lost,
                    "score": tracker.score,
                }, ensure_ascii=False) + "\n")
                log.flush()

                moves.append({
                    "turn": turn,
                    "direction": direction,
                    "frames": frames,
                    "score_before": tracker.score,
                })

                prev_mtime = screen_mtime()
                send_move(direction, frames)

                if not wait_for_turn_done(prev_mtime, timeout=60):
                    print(f"  Game status: {read_status()} — may have stalled.")
                    end_reason = "move_timeout"
                    break

    except KeyboardInterrupt:
        print("\n\nStopped.")
        end_reason = "user_stopped"
    finally:
        (run_dir / "summary.json").write_text(json.dumps({
            "end_reason": end_reason,
            "player": args.player,
            "moves_played": len(moves),
            "start_lives": tracker.start_lives,
            "final_lives": tracker.lives,
            "lives_lost": tracker.lives_lost,
            "final_score": tracker.score,
            "moves": moves,
        }, indent=2), encoding="utf-8")

        print(f"\nRun ended ({end_reason}) after {len(moves)} moves.")
        if tracker.score is not None:
            print(f"Final score: {tracker.score}")
        print(f"Log: {log_path}")
        print(f"Summary: {run_dir / 'summary.json'}")

        print("Sending quit...")
        CMD_FILE.write_text("quit\n")
        time.sleep(2)
        subprocess.run(["pkill", "-f", "Mesen"], capture_output=True)
        subprocess.run(["osascript", "-e", 'quit app "Mesen"'], capture_output=True)
        print("Done.")


if __name__ == "__main__":
    main()

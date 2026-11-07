"""Automated Ms. Pac-Man AI player using mlx-vlm (Apple Silicon GPU).

Launches Mesen, waits for boot, then loops:
    screenshot -> ask local vision LLM -> send move.

Usage:
    python pacman_brain.py
    python pacman_brain.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    python pacman_brain.py --debug              # keep one screenshot per turn
    python pacman_brain.py --baseline random    # no LLM, random moves (benchmark)
    python pacman_brain.py --turns 30

Every run writes output/runs/<timestamp>/turns.jsonl with the raw model
reply, parsed decision and timings for each turn.

Requirements:
    pip install mlx-vlm
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_image
except ImportError:
    sys.exit(
        "mlx-vlm not installed.\n"
        "Install with:  pip install mlx-vlm\n"
        "Requires macOS with Apple Silicon (M1/M2/M3/M4)."
    )

# ---------------------------------------------------------------------------
# Paths — all derived from this script's location and standard Mac locations
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent  # tools/mesen/pacman_mac -> repo root

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
DEFAULT_MODEL = "vikhyatk/moondream2"

KNOWN_MODELS = [
    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "mlx-community/pixtral-12b-4bit",
    "mlx-community/SmolVLM-Instruct-4bit",
    "mlx-community/gemma-3-4b-it-4bit",
    "vikhyatk/moondream2",
    "mlx-community/moondream2-bf16",
    "mlx-community/paligemma2-3b-pt-448-4bit",
    "mlx-community/llava-v1.6-mistral-7b-4bit",
]

MESEN_DATA = Path.home() / "Library" / "Application Support" / "MesenCE"
SCRIPT_DATA = MESEN_DATA / "LuaScriptData" / "pacman_director"
CMD_FILE = SCRIPT_DATA / "command.txt"
STATUS_FILE = SCRIPT_DATA / "output" / "status.txt"
SCREEN_FILE = SCRIPT_DATA / "output" / "screen.png"
WRAM_FILE = SCRIPT_DATA / "output" / "wram.bin"
LOCAL_OUTPUT = SCRIPT_DIR / "output"
RUNS_DIR = LOCAL_OUTPUT / "runs"

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
VALID_DIRECTIONS = ("up", "down", "left", "right")
VALID_FRAMES = (60, 90)
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

LIVES_ADDR = 0x028C
SCORE_ADDR = 0x02A2
SCORE_BYTES = 3
SCORE_BCD = True
SCORE_MULT = 1

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


FALLBACK = {"direction": "left", "frames": 45}


# ---------------------------------------------------------------------------
# WRAM tracking
# ---------------------------------------------------------------------------
def decode_value(snap: bytes, addr: int, width: int, bcd: bool) -> int | None:
    raw = snap[addr : addr + width]
    if len(raw) < width:
        return None
    if bcd:
        digits = "".join(f"{b:02x}" for b in reversed(raw))
        return int(digits) if digits.isdigit() else None
    return int.from_bytes(raw, "little")


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

        value = decode_value(snap, SCORE_ADDR, SCORE_BYTES, SCORE_BCD)
        if value is not None:
            value *= SCORE_MULT
            if self.score is not None and value != self.score:
                events.append(f"score {self.score} -> {value} ({value - self.score:+d})")
            self.score = value
        return events

    def summary(self) -> str:
        score = self.score if self.score is not None else "n/a"
        return f"Lives: {self.lives} | Lost: {self.lives_lost} | Score: {score}"


# ---------------------------------------------------------------------------
# IPC helpers
# ---------------------------------------------------------------------------
def clean_output() -> None:
    for folder in (SCRIPT_DATA / "output", LOCAL_OUTPUT):
        if folder.exists():
            for png in folder.glob("*.png"):
                png.unlink(missing_ok=True)
    STATUS_FILE.unlink(missing_ok=True)
    CMD_FILE.unlink(missing_ok=True)
    WRAM_FILE.unlink(missing_ok=True)


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


# ---------------------------------------------------------------------------
# LLM interaction
# ---------------------------------------------------------------------------
def snap_frames(value: int) -> int:
    return min(VALID_FRAMES, key=lambda f: abs(f - value))


_DIR_SYNONYMS = {
    "up": "up", "north": "up", "top": "up", "upward": "up", "upwards": "up",
    "down": "down", "south": "down", "bottom": "down", "downward": "down", "downwards": "down",
    "left": "left", "west": "left",
    "right": "right", "east": "right",
}
_ALL_DIR_WORDS = "|".join(_DIR_SYNONYMS.keys())
_DIR_RE = re.compile(r"\b(" + _ALL_DIR_WORDS + r")\b", re.IGNORECASE)
_MOVE_RE = re.compile(
    r"\b(?:move|go|head|turn|direction)\s*:?\s*(" + _ALL_DIR_WORDS + r")\b", re.IGNORECASE
)
_PHRASAL_UP = re.compile(
    r"\b(?:pick|set|line|give|end|open|show|speed|catch|wrap|clean|free|sum|"
    r"back|hook|load|sign|fire|power|level|warm|start|mix|round|close|follow|look|come)\s+up\b",
    re.IGNORECASE,
)


def _normalize_dir(raw: str) -> str:
    """Map any direction synonym to up/down/left/right."""
    return _DIR_SYNONYMS.get(raw.lower(), raw.lower())


def parse_decision(text: str) -> list[dict]:
    """Parse: return a list of 1 or 2 moves."""
    frames = FALLBACK["frames"]

    # Priority 1: "GO: ..." line — extract first direction word from it
    go_match = re.search(r"go:\s*(.+)", text, re.IGNORECASE)
    if go_match:
        go_text = go_match.group(1)
        dm = _DIR_RE.search(go_text)
        if dm:
            return [{"direction": _normalize_dir(dm.group(1)), "frames": frames}]

    # Priority 2: "MOVE: direction"
    m = re.search(r"move:\s*(?:go\s+)?(" + _ALL_DIR_WORDS + r")", text, re.IGNORECASE)
    if m:
        return [{"direction": _normalize_dir(m.group(1)), "frames": frames}]

    # Priority 3: JSON with direction field
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            d = _normalize_dir(str(data.get("direction", "")))
            if d in VALID_DIRECTIONS:
                return [{"direction": d, "frames": frames}]
        except json.JSONDecodeError:
            pass

    # Priority 4: "go/move/head/turn right" etc.
    move_matches = list(_MOVE_RE.finditer(text))
    if move_matches:
        return [{"direction": _normalize_dir(move_matches[-1].group(1)), "frames": frames}]

    # Priority 5: last whole-word direction, filtering phrasal "up"
    phrasal_spans = {(m.start(), m.end()) for m in _PHRASAL_UP.finditer(text)}

    direction = FALLBACK["direction"]
    lower = text.lower()
    last_pos = -1
    for m in _DIR_RE.finditer(lower):
        if any(s <= m.start() <= e for s, e in phrasal_spans):
            continue
        if m.start() > last_pos:
            last_pos = m.start()
            direction = _normalize_dir(m.group(1))

    return [{"direction": direction, "frames": frames}]


def ask_model(model, processor, image_path: Path, preprocess: bool = True) -> tuple[list[dict], str]:
    """Returns (move_list, image_sent) where image_sent is the path actually sent to the LLM."""
    used_schematic = False
    if preprocess:
        schematic_path = image_path.parent / "screen_schematic.png"
        try:
            import cv2
            from pacman_vision import analyze_screenshot
            _, _, schematic = analyze_screenshot(str(image_path))
            cv2.imwrite(str(schematic_path), schematic)
            send_path = schematic_path
            used_schematic = True
        except Exception as e:
            print(f"  OpenCV preprocessing failed ({e}), falling back to raw")
            send_path = image_path
    else:
        send_path = image_path

    image_sent = str(send_path)

    try:
        image = load_image(image_sent)
    except Exception as e:
        print(f"  Failed to load image: {e}")
        return [FALLBACK], image_sent

    raw_prompt = SCHEMATIC_PROMPT if used_schematic else RAW_PROMPT
    config = model.config if hasattr(model, "config") else {}
    prompt = apply_chat_template(processor, config, raw_prompt, num_images=1)
    try:
        result = generate(
            model,
            processor,
            prompt,
            image,
            max_tokens=150,
            temperature=0.3,
            verbose=False,
        )
    except Exception as e:
        print(f"  Model generate failed: {e}")
        return [FALLBACK], image_sent

    if result is None:
        print("  Model returned None")
        return [FALLBACK], image_sent
    text = result.text if hasattr(result, "text") else str(result)
    print(f"  Image: {send_path}")
    print(f"  Prompt: {raw_prompt[:80]}...")
    print(f"  Raw response: {repr(text)}")
    return parse_decision(text), image_sent


def random_decision() -> dict:
    return {
        "direction": random.choice(VALID_DIRECTIONS),
        "frames": random.choice(VALID_FRAMES),
        "reasoning": "random baseline",
        "ghost": "none",
        "raw": "",
    }


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Ms. Pac-Man AI Brain (mlx-vlm)")
    parser.add_argument("--mesen", default=str(DEFAULT_MESEN), help="Path to Mesen.app")
    parser.add_argument("--rom", default=str(DEFAULT_ROM), help="Path to ROM file")
    parser.add_argument("--save", default=str(DEFAULT_SAVE), help="Path to save state (.mss)")
    parser.add_argument("--lua", default=str(DEFAULT_LUA), help="Path to pacman_director.lua")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help="HuggingFace model ID. Known models: " + ", ".join(
            m.split("/")[-1] for m in KNOWN_MODELS
        ),
    )
    parser.add_argument("--turns", type=int, default=0, help="Max turns (0=unlimited)")
    parser.add_argument(
        "--baseline",
        choices=("llm", "random"),
        default="llm",
        help="random = skip the LLM and play random moves (benchmark)",
    )
    parser.add_argument(
        "--keep-playing",
        action="store_true",
        help="Do not stop when a life is lost",
    )
    parser.add_argument("--debug", action="store_true", help="Keep one screenshot per turn")
    parser.add_argument(
        "--raw", action="store_true",
        help="Send raw screenshots to the model (skip OpenCV preprocessing)",
    )
    args = parser.parse_args()

    mesen_app = Path(args.mesen)
    rom = Path(args.rom)
    save = Path(args.save)
    lua = Path(args.lua)

    if not mesen_app.exists():
        sys.exit(f"Mesen not found: {mesen_app}")
    if not rom.exists():
        sys.exit(f"ROM not found: {rom}")
    if not save.exists():
        sys.exit(f"Save state not found: {save}\n  Create one in Mesen: load the game, get to gameplay, then Save State > Slot 1")
    if not lua.exists():
        sys.exit(f"Lua script not found: {lua}")

    # Load the vision model (downloads on first run)
    model_obj = None
    processor_obj = None
    if args.baseline == "llm":
        print("Loading model...")
        print("  (First run downloads the model — may take a few minutes)")
        try:
            model_obj, processor_obj = load(args.model)
        except Exception as e:
            sys.exit(
                f"Failed to load model: {e}\n\n"
                f"Try one of these models:\n"
                f"  --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit\n"
                f"  --model mlx-community/paligemma2-3b-pt-448-4bit\n"
                f"  --model mlx-community/llava-v1.6-mistral-7b-4bit\n"
                f"  --model mlx-community/moondream2-bf16\n"
            )

        print("Model loaded!")

    SCRIPT_DATA.mkdir(parents=True, exist_ok=True)
    (SCRIPT_DATA / "output").mkdir(parents=True, exist_ok=True)
    clean_output()

    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "turns.jsonl"

    mode = "raw" if args.raw else "schematic"
    label = "random baseline" if args.baseline == "random" else f"{args.model} ({mode})"
    tracker = GameTracker()
    print(f"\nPlayer: {label}")
    print(f"Run log: {log_path}")

    print("Launching Mesen...")
    subprocess.Popen(["open", str(mesen_app), "--args", str(rom), str(save), str(lua)])

    print("Waiting for game to load...")
    if not wait_for_status("waiting", timeout=120):
        print("ERROR: Game did not reach waiting state.")
        print("  Check Mesen Script Window > Settings:")
        print("  - 'Allow I/O and OS access' must be enabled")
        subprocess.run(["pkill", "-f", "Mesen"], capture_output=True)
        return

    print("Game ready! AI is playing...\n")
    turn = 0
    last = "none yet"
    recent_moves: list[str] = []
    moves: list[dict] = []
    end_reason = "unknown"

    try:
        with open(log_path, "w", encoding="utf-8") as log:
            while True:
                turn += 1
                if args.turns and turn > args.turns:
                    print(f"Reached {args.turns} turn limit.")
                    end_reason = "turn_limit"
                    break

                screen = grab_screenshot()
                if not screen:
                    print("  No screenshot, waiting...")
                    time.sleep(2)
                    turn -= 1
                    continue

                snap = read_wram()
                if snap:
                    for event in tracker.update(snap):
                        print(f"  *** {event}")
                    print(f"  {tracker.summary()}")
                    if tracker.lives_lost > 0 and not args.keep_playing:
                        print("  Life lost — stopping run.")
                        log.write(
                            json.dumps(
                                {
                                    "event": "life_lost",
                                    "time": datetime.now().isoformat(timespec="seconds"),
                                    "lives": tracker.lives,
                                    "lives_lost": tracker.lives_lost,
                                    "score": tracker.score,
                                }
                            )
                            + "\n"
                        )
                        log.flush()
                        end_reason = "life_lost"
                        break

                if args.debug:
                    shutil.copy2(screen, run_dir / f"turn_{turn:03d}.png")

                print(f"Turn {turn}: thinking...")
                t_llm = time.time()
                if args.baseline == "random":
                    move_list = [random_decision()]
                    image_sent = str(screen)
                else:
                    move_list, image_sent = ask_model(model_obj, processor_obj, screen, preprocess=not args.raw)
                llm_secs = time.time() - t_llm
                if args.baseline == "llm":
                    print(f"  LLM took {llm_secs:.1f}s")

                move_names = [m["direction"] for m in move_list]
                print(f"-> {' -> '.join(move_names)} ({len(move_list)} move{'s' if len(move_list) > 1 else ''})")

                move_failed = False
                for mi, move in enumerate(move_list):
                    direction = move["direction"]
                    frames = move["frames"]

                    turn_entry = {
                                "turn": turn,
                                "step": mi + 1,
                                "time": datetime.now().isoformat(timespec="seconds"),
                                "llm_seconds": round(llm_secs, 2) if mi == 0 else 0,
                                "direction": direction,
                                "frames": frames,
                                "lives": tracker.lives,
                                "lives_lost": tracker.lives_lost,
                                "score": tracker.score,
                    }
                    if mi == 0:
                        turn_entry["image_sent"] = image_sent
                    log.write(json.dumps(turn_entry, ensure_ascii=False) + "\n")
                    log.flush()

                    moves.append(
                        {
                            "turn": turn,
                            "direction": direction,
                            "frames": frames,
                            "score_before": tracker.score,
                        }
                    )
                    last = f"{direction} for {frames} frames"
                    recent_moves.append(direction)
                    prev_mtime = screen_mtime()
                    send_move(direction, frames)

                    done = wait_for_turn_done(prev_mtime, timeout=60)
                    if not done:
                        print(f"  Game status: {read_status()}")
                        print("  Game may have ended or stalled.")
                        end_reason = "move_timeout"
                        move_failed = True
                        break

                if move_failed:
                    break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        end_reason = "user_stopped"
    finally:
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "end_reason": end_reason,
                    "player": label,
                    "moves_played": len(moves),
                    "start_lives": tracker.start_lives,
                    "final_lives": tracker.lives,
                    "lives_lost": tracker.lives_lost,
                    "final_score": tracker.score,
                    "moves": moves,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nRun ended ({end_reason}) after {len(moves)} moves. {tracker.summary()}")
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

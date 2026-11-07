"""Send commands to the Ms. Pac-Man director manually.

Mac version — paths point to ~/Library/Application Support/MesenCE.

Usage:
    python pacman_cmd.py status
    python pacman_cmd.py screenshot
    python pacman_cmd.py move right 60
    python pacman_cmd.py quit
    python pacman_cmd.py wait          # block until game reaches "waiting"
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

MESEN_DATA = Path.home() / "Library" / "Application Support" / "MesenCE"
SCRIPT_DATA = MESEN_DATA / "LuaScriptData" / "pacman_director"
CMD_FILE = SCRIPT_DATA / "command.txt"
STATUS_FILE = SCRIPT_DATA / "output" / "status.txt"
SCREEN_FILE = SCRIPT_DATA / "output" / "screen.png"
LOCAL_SCREEN = Path(__file__).resolve().parent / "output" / "screen.png"


def read_status() -> str:
    if STATUS_FILE.exists():
        return STATUS_FILE.read_text().strip()
    return "unknown"


def send_move(direction: str, frames: int, delay: int = 0) -> str:
    if delay:
        time.sleep(delay)
    CMD_FILE.write_text(f"{direction} {frames}\n")
    for _ in range(60):
        time.sleep(0.5)
        if read_status() == "waiting":
            break
    return grab_screenshot()


def grab_screenshot() -> str:
    if SCREEN_FILE.exists():
        LOCAL_SCREEN.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SCREEN_FILE, LOCAL_SCREEN)
        return str(LOCAL_SCREEN)
    return "no screenshot"


def wait_for_ready(timeout: int = 90) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if read_status() == "waiting":
            return True
        time.sleep(1)
    return False


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        print(f"Status: {read_status()}")
    elif args[0] == "status":
        print(f"Status: {read_status()}")
    elif args[0] == "screenshot":
        print(grab_screenshot())
    elif args[0] == "quit":
        CMD_FILE.write_text("quit\n")
        print("Quit sent.")
    elif args[0] == "move" and len(args) >= 3:
        delay = int(args[3]) if len(args) > 3 else 0
        result = send_move(args[1], int(args[2]), delay)
        print(f"Screenshot: {result}")
    elif args[0] == "wait":
        ok = wait_for_ready(int(args[1]) if len(args) > 1 else 90)
        print("ready" if ok else "timeout")
    else:
        print(f"Unknown: {args}")

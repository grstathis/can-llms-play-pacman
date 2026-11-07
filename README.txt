Ms. Pac-Man AI Player — Mac + mlx-vlm (Apple Silicon)
=====================================================

Runs a local vision LLM on your Mac's GPU (M1/M2/M3/M4) to play
Ms. Pac-Man autonomously. No cloud API, no Ollama — just Python
and Apple Silicon.

FILES
-----
pacman_director.lua  — Lua script for Mesen. Freezes the emulator
                       between moves via savestate reload. Dumps WRAM
                       for score/lives tracking.

pacman_brain.py      — Automated AI player. Launches Mesen, loads
                       a vision model, plays turn by turn.

pacman_cmd.py        — Manual command sender for interactive play.

SETUP
-----
1. Mesen (MesenCE) must be in the repo root as Mesen.app.
   Open it once, go to Tools > Script Window > Settings,
   enable "Allow I/O and OS access", then close it.

2. ROM at: <repo>/rom/Ms. Pac-Man (USA).sfc

3. Create a save state in Mesen:
   - Load the ROM, get past the title screen to actual gameplay
   - File > Save State > Slot 1
   - The save state goes to:
     ~/Library/Application Support/MesenCE/SaveStates/Ms. Pac-Man (USA)_1.mss

4. Install Python dependencies:
       pip install mlx-vlm

   This pulls in MLX, transformers, and everything needed.
   First model download happens on the first run (~2-4 GB).

AUTOMATED PLAY
--------------
From the pacman_mac folder:

    python pacman_brain.py

Options:
    --model MODEL    HuggingFace model ID (default: Qwen2.5-VL-3B)
    --turns N        Stop after N turns (0 = unlimited)
    --baseline random    Random moves, no LLM (for benchmarking)
    --keep-playing       Don't stop on first life lost
    --debug              Save a screenshot per turn in the run folder

RECOMMENDED MODELS (16GB Mac)
-----------------------------
Small & fast:
    --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit    (~2 GB)
    --model mlx-community/moondream2-bf16                  (~3.5 GB)
    --model mlx-community/paligemma2-3b-pt-448-4bit        (~2 GB)

Larger & smarter:
    --model mlx-community/llava-v1.6-mistral-7b-4bit       (~4 GB)
    --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit     (~4 GB)

If a model doesn't work, try another — mlx-vlm adds new models often.
Check: https://huggingface.co/mlx-community?search=vlm

MANUAL PLAY
-----------
Step 1: Launch Mesen manually:
    open ../../../Mesen.app --args "$(pwd)/../../../rom/Ms. Pac-Man (USA).sfc" \
        "$HOME/Library/Application Support/MesenCE/SaveStates/Ms. Pac-Man (USA)_1.mss" \
        "$(pwd)/pacman_director.lua"

Step 2: Check status:
    python pacman_cmd.py status
    -> should say "waiting"

Step 3: Grab a screenshot:
    python pacman_cmd.py screenshot

Step 4: Send a move:
    python pacman_cmd.py move right 60

Step 5: Repeat from Step 3.

To quit:
    python pacman_cmd.py quit

HOW IT WORKS
------------
1. Mesen loads a save state (skips title screen entirely).
2. The Lua script creates an in-memory savestate and reloads it every
   frame — the game is completely frozen while the AI thinks.
3. The Lua script writes a screenshot + WRAM dump to disk.
4. The Python script reads the screenshot, sends it to the local LLM.
5. The LLM returns a direction, the Python script writes it to command.txt.
6. The Lua script reads the command, plays the move for N frames, then
   freezes again.

60 frames = 1 second of gameplay. Shorter moves (30-45) give tighter
control but cost more LLM calls.

RUN LOGS
--------
Every run saves to output/runs/<timestamp>/:
    turns.jsonl   — one JSON line per turn (direction, score, timing)
    summary.json  — final stats (score, lives, end reason)
    turn_NNN.png  — per-turn screenshots (only with --debug)

TIPS
----
- The maze scrolls — it's larger than the screen. What looks like a new
  level might just be a different part of the same maze.
- Ghost strategy: move AWAY from ghost clusters, sweep the far side.
- Power pellets (big dots) let you eat ghosts for 200 points each.
- The ghost avoidance rule automatically flips your direction if the
  model tries to move toward the nearest ghost.

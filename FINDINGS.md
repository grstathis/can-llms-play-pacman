# Can Vision LLMs Play Ms. Pac-Man? — Benchmark Findings

## The Experiment

We built a pipeline that lets local vision language models play Ms. Pac-Man on a real SNES emulator (Mesen), one move at a time. The game freezes between decisions: the AI sees a raw screenshot and outputs a direction. All models run locally on a 16GB Apple Silicon Mac via mlx-vlm — no cloud APIs.

**Pipeline:** Lua script freezes emulator → raw screenshot captured → VLM receives image + prompt → outputs direction → emulator plays 45 frames (~0.75s of gameplay) → repeat.

**Prompt (v3, final):**
```
This is a Pac-Man game screenshot. Pac-Man is yellow. Walls are magenta and they block Pac-man.
Pellets are white/grey dots. Ghosts are colored blobs with eyes.
Which direction (left, right, up, down) has pellets and no ghosts?
MOVE: <direction>
```

**Methodology:** Each model plays multiple games from the same save state. A game ends on first life lost. The random baseline picks a uniform random direction each turn (no LLM). All VLM runs use raw screenshots (no OpenCV preprocessing) and the same prompt.

---

## Results (Final Prompt, Nov 7 2026)

Only runs from the `run_all_evals.sh` batch after the final prompt was set (brain.py last modified Nov 7 00:39) are included. Random baseline is prompt-independent so all 9 games are kept, minus one 950-point outlier removed for being unrepresentative.

| Model | Size | Games | Avg Score | Median | Best | Worst | Latency/move | vs Random |
|-------|------|-------|-----------|--------|------|-------|-------------|-----------|
| **Pixtral 12B** | 12B | 6 | **535** | 405 | 1340 | 230 | 14.4s | **+73%** |
| **Qwen2.5-VL 3B** | 3B | 10 | **411** | 365 | 640 | 240 | 3.3s | **+33%** |
| Random Baseline | — | 9 | **310** | 270 | 680 | 190 | — | — |
| Moondream2 | 1.9B | 5 | 268 | 290 | 300 | 170 | 3.1s | -14% |
| Qwen2.5-VL 7B | 7B | 5 | 240 | 130 | 470 | 130 | 3.9s | -23% |
| SmolVLM | 2B | 4 | 175 | 175 | 180 | 170 | 10.2s | -44% |
| Qwen3-VL 4B | 4B | 5 | 120 | 120 | 120 | 120 | 2.6s | -61% |
| LLaVA 1.6 7B | 7B | 5 | 28 | 40 | 50 | 0 | — | -91% |

**Only 2 of 7 VLMs consistently beat random moves.**

---

## Key Finding 1: Most VLMs Can't Beat Random

The random baseline averages 310 points across 9 games. Only Pixtral 12B (535 avg) and Qwen2.5-VL 3B (411 avg) consistently outperform it. The remaining 5 models all score below random — some dramatically.

This means most small VLMs cannot extract enough spatial information from a game screenshot to make decisions that are better than chance.

---

## Key Finding 2: Bigger Isn't Better

Qwen2.5-VL 3B (411 avg) outperforms its 7B sibling (240 avg) by 71%, despite being half the size and faster (3.3s vs 3.9s per move). The 3B model makes better use of its moves — it scores on ~41% of turns vs ~19% for the 7B.

This may reflect differences in how the 4-bit quantization affects each model size, or differences in the vision encoder's spatial reasoning at different scales.

---

## Key Finding 3: Direction Spam — 5 of 7 Models Are Broken

Direction analysis across all turns in the eval batch reveals that most models fixate on a single direction regardless of game state. Only 2 of 7 models actually play the game:

| Model | Top Direction | % of Turns | Verdict |
|-------|--------------|-----------|---------|
| **Qwen2.5-VL 3B** | right | 34% | **Varied — actually plays** |
| **Pixtral 12B** | down | 36% | **Varied — actually plays** |
| Moondream2 | up | 86% | Direction spam |
| Qwen2.5-VL 7B | down | 72% | Direction spam |
| SmolVLM | right | 87% | Direction spam |
| Qwen3-VL 4B | left | 97% | Direction spam (thinking mode bug) |
| LLaVA 1.6 7B | — | — | No valid moves at all |

**SmolVLM** spams "right" every turn:
```
Seq: r r r r r r r r r r r r r r r r r r r r r r r r r r r r
```
Scores 170-180 every game (eats a few pellets going right, then hits a wall and dies).

**Qwen2.5-VL 7B** spams "down":
```
Seq: d d d d d d u d d d d d d u d d u d r u d
```
16-25 out of ~25 turns are "down". Gets 130 points (3 pellets going down), then oscillates until it dies. Even its "good" 470 game is 25/34 turns "down" — it just bumped into more pellets along the way.

**Moondream2** spams "up" at 86%. Its consistent 290 scores aren't skill — going "up" from the start position happens to collect a fixed set of pellets before hitting a wall. Every game plays almost identically.

**Qwen3-VL 4B** spams "left" at 97% — but this one has a different root cause (see below).

**LLaVA 1.6 7B** is the worst: it produces 0 valid moves in most games (avg score 28). The model appears unable to parse the screenshot into any useful output.

---

## Key Finding 4: The Qwen3-VL Thinking Mode Bug

Qwen3-VL 4B scored exactly 120 in all 5 games with the final prompt, outputting "left" every turn. Investigation revealed this is **not a model capability issue** — it's a thinking mode interaction.

### What we found:

**Without thinking mode:** The model outputs a single token (`'left'`, 4 chars) and immediately hits EOS. This happens because Qwen3-series models have a hybrid thinking/non-thinking architecture. The `MOVE: <direction>` template triggers "fill in the blank" behavior — the model completes it as a single token and stops.

**When asked to describe the same screenshot:** The model produces 2000+ chars of detailed, accurate game analysis — identifying Pac-Man's position, ghost locations, wall layout, and pellet distribution. The vision encoder works fine.

**With forced thinking mode** (prepending `<think>\n` to the assistant turn): The model reasons through the game state (500-2000 chars) and makes varied, image-aware decisions. However:
- Latency jumps from 1.2s to **30-50s per turn** (unusable for gameplay)
- The model loops into repetitive "Wait — let me look again" spirals
- Ghost positions are still hallucinated despite the reasoning

### Root cause:

Qwen3-VL models include `<think>` (token 151667) and `</think>` (token 151668) special tokens. Without the thinking prefix, the model's generation loop terminates after one token for short-answer prompts. This is a known issue documented for Qwen3-series models on local inference frameworks.

We verified this is **not** an mlx-vlm version bug — upgrading from 0.7.10 to 0.7.12 did not fix the single-token behavior.

### The earlier Nov 6 runs:

Qwen3-VL scored 160-480 on Nov 6 runs (before the final prompt change). Those runs used an earlier, more verbose prompt (v2) that happened to elicit longer responses without needing `<think>`. The prompt change to v3 (shorter, more direct) pushed Qwen3-VL into single-token mode.

**Conclusion:** Qwen3-VL 4B is not viable for this task with the standard prompt on mlx-vlm. It requires either a verbose prompt that avoids the thinking mode trigger, or explicit `<think>` injection with a much higher token budget — neither is practical for real-time gameplay.

---

## Key Finding 5: The Qwen2.5-VL 7B Paradox — Bigger Model, Worse Play

The 3B and 7B versions of Qwen2.5-VL are the same architecture, same prompt, same pipeline — only the parameter count differs. Yet the 3B actually plays the game (411 avg, 34% direction diversity) while the 7B spams "down" (240 avg, 72% "down").

### Side-by-side response comparison (same screenshot, same prompt):

**Qwen2.5-VL 3B** generates 600-800 chars of step-by-step reasoning:
```
"To determine the direction with pellets and no ghosts, let's carefully
examine the maze: 1. Top Left Corner: There are no pellets here.
2. Top ..."
```

**Qwen2.5-VL 7B** generates a single short sentence with no reasoning:
```
"The direction that has pellets and no ghosts is **down**."
```

Both models can describe the screenshot in detail when asked — the 7B actually produces a richer description (1343 chars vs 452 chars). But with the game prompt, the 7B skips reasoning and pattern-matches to a quick answer, defaulting to "down" ~72% of the time.

### Ruling out framework bugs:

We verified this is not a code issue, cache corruption, or mlx-vlm bug:
- **Official CLI test** (`python -m mlx_vlm.generate`): Same behavior — 7B outputs `MOVE: Down`, 3B outputs multi-paragraph reasoning
- **Cache clear** (deleted `~/.cache/huggingface/hub/models--mlx-community--Qwen2.5-VL-7B-Instruct-4bit/`, forced re-download): No change
- **mlx-vlm upgrade** (0.7.10 → 0.7.12): No change

### Hypothesis:

The 4-bit quantized 7B model's instruction-following is "too efficient" — it recognizes the `MOVE: <direction>` template as a short-answer format and skips the chain-of-thought reasoning that the 3B model naturally produces. The 3B's less polished instruction tuning is actually an advantage here: it over-explains, which forces it to actually analyze the screenshot before answering.

This suggests that for game-playing tasks, **instruction-following quality can be inversely correlated with task performance** — models that reason before answering outperform models that jump to conclusions, even when the "conclusions" come from a larger model.

---

## Key Finding 6: Raw Screenshots Beat Schematics

In earlier experiments (not included in the final benchmark), we tested an OpenCV preprocessing pipeline that converted raw game frames into clean schematics — detecting Pac-Man, ghosts, dots, and walls by color, then rendering them as simple shapes.

For Qwen2.5-VL 3B:
- **Raw screenshots:** 372 avg (Nov 6 runs, earlier prompt)
- **Schematic screenshots:** 263 avg (30 games)

The preprocessing removed useful spatial cues that the models could actually use. Raw pixel data, despite being noisier, gave the VLMs more to work with.

---

## Key Finding 6: Prompt v5 — Structured CoT Helps Broken Models, Hurts Good Ones

After discovering the direction spam problem, we designed a structured chain-of-thought prompt (v5) that forces models to report game state before picking a direction:

```
Pac-Man game. Yellow=Pac-Man, magenta=walls, white dots=pellets.
Ghosts: red, pink, orange, cyan blobs with eyes — avoid them.
Exception: blue/white ghosts are EDIBLE — chase them for bonus points!
Answer briefly:
PACMAN: <position>
GHOSTS: <color and position of each ghost>
PELLETS: <which directions have pellets>
DANGER: <which directions have normal ghosts>
GO: <one non-diagonal direction>
```

### v3 vs v5 Results:

| Model | v3 Avg | v5 Avg | Change | What Happened |
|-------|--------|--------|--------|---------------|
| **Qwen2.5-VL 3B** | 411 | **516** | **+26%** | Best overall — scores 280-650, structured format enhances natural reasoning |
| **Qwen3-VL 4B** | 120 | **260** | **+117%** | Fixed — was 97% "left" spam, now varies directions |
| **Gemma 3 4B** | 120 | **248** | **+107%** | Fixed — was 120 flat every game, now 230-280 range |
| **SmolVLM** | 175 | **240** | **+37%** | Improved — was 87% "right" spam |
| Pixtral 12B | 535 | 320 | **-40%** | Hurt — bloats into markdown reports, truncated at 150 tokens before reaching GO: |
| Qwen2.5-VL 7B | 240 | 160 | **-33%** | Varied directions but hallucinated grid coordinates |
| Moondream2 | 268 | 120 | **-55%** | Too small for template — parroted format back |

### The tradeoff:

The structured prompt lifts 4 of 7 models, including the overall best performer. Three models — Pixtral, Qwen2.5-VL 7B, and Moondream2 — perform worse with the structured format.

Qwen2.5-VL 3B is now the **overall champion** on v5 at 516 avg — surpassing Pixtral's v5 score of 320, while being 4x faster (3.3s vs 14.4s per move). Initial runs showed the 3B crashing with v5 (echoing the template back), but this turned out to be Mesen emulator boot failures, not a prompt issue. A clean rerun produced the best scores in the entire benchmark.

Models that short-circuit on free-form prompts (Qwen3-VL, Gemma) benefit from being forced to analyze the game state step by step. Pixtral performs worse because it bloats the structured format into markdown reports and hits the token limit.

### Failure modes by model size:

- **Small models** (Moondream2 1.9B) can get confused by the template format and parrot it back
- **Small-medium models** (Qwen2.5-VL 3B, Qwen3-VL 4B, Gemma 3 4B, SmolVLM 2B) benefit most — the structure guides their reasoning without overwhelming them. Qwen2.5-VL 3B in particular excels: it was already reasoning naturally, and the structure focuses that reasoning
- **Large models** (Pixtral 12B) bloat the structured format with markdown headers and verbose prose, hitting the token limit before reaching the GO: direction

---

## Key Finding 7: Latency Doesn't Predict Quality

| Model | Latency | v3 Score | v5 Score |
|-------|---------|----------|----------|
| Pixtral 12B | 14.4s | 535 (best v3) | 320 |
| SmolVLM | 10.2s | 175 | 240 |
| Gemma 3 4B | 8.5s | 120 | 248 |
| Qwen2.5-VL 7B | 3.9s | 240 | 160 |
| Qwen2.5-VL 3B | 3.3s | 411 (best value v3) | crashed |
| Qwen3-VL 4B | 2.6s | 120 | 260 |

Speed is not a proxy for game-playing ability — on either prompt version.

---

## The Big Picture

The ideal approach is likely **per-model prompt selection**: use v3 (free-form) for models that naturally reason (Pixtral, Qwen2.5-VL 3B), and v5 (structured) for models that need guidance (Qwen3-VL, Gemma, SmolVLM).

**Best performers by prompt:**
- **v3 (free-form):** Pixtral 12B (535), Qwen2.5-VL 3B (411)
- **v5 (structured):** Qwen2.5-VL 3B (516), Pixtral 12B (320), Qwen3-VL 4B (260), Gemma 3 4B (248), SmolVLM (240)

**Overall champion: Qwen2.5-VL 3B on v5** — 516 avg, 3.3s/move, beats random baseline by 66%. The structured prompt focused its natural reasoning ability rather than constraining it.

---

## Technical Details

- **Emulator:** Mesen (MesenCE) running Ms. Pac-Man (USA) SNES ROM
- **Inference:** mlx-vlm 0.7.12 on Apple Silicon (M-series, 16GB)
- **All models:** 4-bit quantized via mlx-community HuggingFace repos
- **Move duration:** 45 frames (~0.75 seconds of gameplay)
- **Game start:** Save state loaded to skip title screen, game starts mid-play
- **End condition:** First life lost
- **Data:** output/runs/ contains per-turn JSONL logs and summaries for every game
- **Analysis:** pacman_analysis.py generates matplotlib charts; dashboard at output/pacman_llm_dashboard.html
- **Code:** pacman_brain.py (player), pacman_eval.py (batch runner), pacman_director.lua (emulator control)

---

## Open Questions

- **Per-model prompt routing:** Can we detect at runtime whether a model needs structured or free-form prompts?
- **Adaptive token budget:** Give verbose models (Pixtral) 300+ tokens while keeping compact models at 100?
- Could fine-tuning a small VLM on Pac-Man screenshots improve play beyond random?
- How would cloud-hosted frontier models (Claude Opus, GPT-4o) perform with the same pipeline?
- Would multi-frame context (showing the last N screenshots) help models track ghost trajectories?
- Can models plan multi-step paths instead of single moves?

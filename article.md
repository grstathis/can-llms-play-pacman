# Can Vision LLMs Play Ms. Pac-Man?

*What we learned about AI spatial reasoning by letting them play a Pac-Man game.*

[IMAGE: output/screen.png — A raw Ms. Pac-Man screenshot from the SNES emulator. This is exactly what every AI model sees each turn.]

Here's the experiment: take a Ms. Pac-Man game running in an SNES emulator, freeze it between moves, and let a vision AI look at a single screenshot and pick a direction. No memory of previous turns, no feedback on whether a move worked, no game-specific training — just a model, one frame, and a question: *which way should Pac-Man go?*

We tested 7 open-weight vision language models (VLMs) running locally on an M1 MacBook Pro — no cloud APIs, no GPU rental. Vision language models are AI systems that can take both images and text as input — you show them a picture and ask a question about it. The models ranged from 1.9 billion to 12 billion parameters, all running on-device. We later added Claude Opus 4.6 as a remote frontier comparison.

Only 2 of 7 local models scored above the random baseline on average. Most fixated on a single direction — "right" every turn, or "down" every turn — regardless of what was on screen. A random number generator was a better Pac-Man player than five of seven AI models.

But the more interesting story is *why* they failed, how we fixed some of them with nothing but prompt changes, and what happened when a much larger frontier model entered the arena. Along the way, we found some genuinely surprising things about how these models work — and don't work — when they need to reason about space.

---

## The Setup: Freezing an SNES Game for an AI

The game runs inside [Mesen](https://www.mesen.ca/), an SNES emulator. A Lua script inside the emulator freezes the game between every move:

```lua
-- Simplified game loop inside the emulator
function on_each_frame()
    if state == FROZEN then
        reload_checkpoint()          -- keep the current board state frozen
        direction = read_command_file()
        if direction then
            start_playing(direction, 45)  -- play 45 frames (~0.75s)
        end
    elseif state == PLAYING then
        frames_played = frames_played + 1
        if frames_played >= 45 then
            save_checkpoint()        -- snapshot the new board state
            take_screenshot()        -- save PNG to disk
            dump_game_memory()       -- save lives/score
            state = FROZEN           -- freeze until next AI command
        end
    end
end
```

The AI has unlimited thinking time. After each move, the Lua script freezes the game in place — takes a screenshot, then holds everything still until the AI provides its next direction. Ghosts can't move, timers don't tick, nothing changes. Once the AI writes a direction, the script plays 45 frames (~0.75 seconds) and freezes again.

On the Python side: wait for a screenshot, load it into the vision model, parse the response for a direction, write it to a command file, repeat. We read the SNES work RAM directly to track score and lives — the emulator exposes raw game memory, so we know exactly when a life is lost or points are scored without relying on OCR or pixel counting.

The communication between all the pieces is deliberately low-tech: the Lua script writes a PNG file, the Python script reads it. The Python script writes a text file with a direction, the Lua script reads it next frame. No sockets, no RPC, no message queues.

All local models run on an M1 MacBook Pro (16GB) via [mlx-vlm](https://github.com/Blaizzy/mlx-vlm), a framework for running vision language models on Apple Silicon GPUs. Every model is 4-bit quantized — compressed to fit in 2–4 GB of memory. We capped output at 150 tokens per move to keep inference fast; most models use far less, but verbose ones like Pixtral can hit the limit. No cloud APIs, no GPU rental.

The models we tested: **Qwen2.5-VL** (3B and 7B), **Pixtral 12B**, **Qwen3-VL 4B**, **SmolVLM** (2B), **Moondream2** (1.9B), and **Gemma 3 4B** — all 4-bit quantized versions from the mlx-community on HuggingFace. And later, **Claude Opus 4.6** as the frontier comparison.

**Methodology:** Every run starts from the same save state, so all models face identical initial conditions. A run ends on first life lost — so scores represent performance up to first death, not a full playthrough. Each model played 5 runs per prompt version. If a model outputs an invalid direction, the parser falls back to the last valid direction. If a move goes into a wall, the game plays it anyway — Pac-Man just doesn't move, and the score stays the same. All scores reported are averages; individual run scores vary widely (the full dataset includes min/max).

A quick note on the game itself: Ms. Pac-Man for SNES is harder than it might look. The maze is larger than the screen — the view scrolls as Pac-Man moves — so the AI never sees the full layout at once. Unlike the original Pac-Man, which has one fixed maze, Ms. Pac-Man rotates between different maze layouts. The ghosts are fast and unpredictable. And everything is rendered in a very specific SNES art style: magenta walls, black corridors, white pellets, and colored ghost sprites at a native 256×224 resolution. This is what the models have to work with — no high-res renders, no clean vector graphics, just raw 16-bit pixel art.

[VIDEO: A short clip of the AI playing Ms. Pac-Man, showing the freeze-move-freeze cycle in action. Each pause is the model thinking.]

---

## First Results: Most Models Just Spam One Direction

We started with a minimal prompt:

```
This is a Pac-Man game screenshot. Pac-Man is yellow. Walls are magenta
and they block Pac-Man. Pellets are white/grey dots. Ghosts are colored
blobs with eyes.
Which direction (left, right, up, down) has pellets and no ghosts?
MOVE: <direction>
```

Two models actually played the game. **Qwen2.5-VL 3B** — a 3-billion-parameter model from Alibaba's Qwen team — averaged 411 points across 5 runs, varying its directions based on what it saw on screen. **Pixtral 12B** — Mistral's vision model — scored even higher at 535 points. Both showed genuine direction diversity: their most-used direction accounted for only about 35% of moves, meaning they were actually responding to different game states with different decisions.

The other five failed to beat the random baseline.

**SmolVLM** (2B parameters) output "right" on 87% of its turns. Every game looked the same: go right, eat a few pellets in that corridor, hit a wall, die. Score: 170–180, every single time.

```
SmolVLM move sequence: r r r r r r r r r r r r r r r r r r r r r r r r r r r r
```

**Qwen2.5-VL 7B** spammed "down" on 72% of turns. **Moondream2** spammed "up" on 86%. **Qwen3-VL 4B** spammed "left" on an astounding 97% of turns.

For context, we ran a **random baseline** using the same starting state and stopping rule: pick a direction uniformly at random each turn, no AI at all. It averaged **310 points** over 5 runs. In this setup, five of seven models averaged below that baseline.

Each turn is one model decision followed by 45 frames (~0.75s) of gameplay.

| Model | Size | Avg Score | Min–Max | Avg Turns | Latency/move | Top Dir | % Spam |
|-------|------|-----------|---------|-----------|-------------|---------|--------|
| **Pixtral 12B** | 12B | **535** | 230–1340 | 30 | 14.4s | down | 36% |
| **Qwen2.5-VL 3B** | 3B | **411** | 240–640 | 23 | 3.3s | right | 34% |
| Random Baseline | — | **310** | 190–680 | 14 | — | — | — |
| Moondream2 | 1.9B | 268 | 170–300 | 14 | 3.1s | up | 86% |
| Qwen2.5-VL 7B | 7B | 240 | 130–470 | 28 | 3.9s | down | 72% |
| SmolVLM | 2B | 175 | 170–180 | 27 | 10.2s | right | 87% |
| Qwen3-VL 4B | 4B | 120 | 120–120 | 22 | 2.6s | left | 97% |
| Gemma 3 4B | 4B | 120 | 120–120 | 22 | 8.5s | — | — |

Moondream2's scores cluster around 290 in most runs — which *looks* like skill, but going "up" from the start position just happens to collect a fixed set of pellets before hitting a wall. Every game plays out almost identically. The model isn't navigating; it's accidentally replaying the same path.

Because the maze is larger than the screen and scrolls as Pac-Man moves, a model that spams one direction can actually travel quite far before hitting a dead end — and along the way, it eats 10–20 pellets purely by accident. That's why even the worst spammers don't score zero.

The turns column tells its own story. SmolVLM and Qwen2.5-VL 7B survive 27–28 turns despite low scores — they're alive but not collecting pellets, consistent with repeatedly choosing blocked directions. Moondream2 and the random baseline both die in ~14 turns, but Moondream2 scores nearly as much by spamming "up" as random does by varying directions — a reminder that survival time and score don't always correlate.

---

## Why the Models Break

A major contributor to the direction-spam pattern was our prompt's last line: `MOVE: <direction>`. For instruction-tuned models, this looks like a fill-in-the-blank template. They complete it with a single token and stop — no visible reasoning or evidence of image analysis.

**The Qwen3-VL think tag:** Qwen3-VL scored exactly 120 in every run, 97% "left." But when asked to simply *describe* the same screenshot, it produced 2,000+ characters of accurate game analysis. The vision encoder worked fine.

In our mlx-vlm setup, Qwen3-VL's generation terminated after one token for short-answer prompts. Qwen3 models have special `<think>` and `</think>` tokens that gate their reasoning mode, and without an explicit `<think>` prefix, the `MOVE: <direction>` template triggered single-token completion — the model filled in one word and stopped.

We confirmed it by prepending `<think>\n` to force reasoning. The model produced 600+ characters of spatial analysis and picked varied directions — but at 30–50 seconds per turn, looping into "Wait — let me look again" spirals. The reasoning was there; the prompt just wasn't triggering it. This is a gotcha worth knowing about for anyone using Qwen3 models locally: if you're getting suspiciously short answers, the think tag might be the issue.

**The 3B beats 7B paradox:** Same model family, same prompt, same screenshot — yet the 3B produces 623 characters of step-by-step reasoning while the 7B outputs just "The direction that has pellets and no ghosts is **down**." (57 characters). The 7B's instruction-following is *too efficient*. It recognizes the short-answer format and skips thinking entirely. The 3B's less polished tuning is an advantage — it over-explains, which forces it to actually analyze the screenshot.

**In this setup, bigger and better-tuned models were more susceptible** to this pattern. They're better at recognizing short-answer formats, so they skip straight to the answer. At least for this kind of task, instruction-following quality can work against performance.

---

## Fixing It: From `MOVE:` to `GO:`

Once we understood that prompt structure was a major contributor to the direction spam — alongside wall perception and other factors — we started iterating. Our first fix was simple: add "say why in one sentence" before `MOVE: direction`. This helped in isolated tests — Qwen3-VL and the 7B both started producing varied, reasoned responses on individual screenshots. But in full games, Qwen3-VL still only averaged 144 points. One sentence of reasoning wasn't enough to consistently break the directional bias turn after turn. The model would reason "there are pellets to the left" and then output "left" ten turns in a row regardless.

The breakthrough came with a different approach — **a structured prompt** that asks the model to identify positions, hazards, and pellet directions before choosing a move:

```
This is a Pac-Man game. Yellow=Pac-Man, magenta=walls, white dots=pellets.
Ghosts: red, pink, orange, cyan blobs with eyes — avoid them.
Exception: blue/white ghosts are EDIBLE — chase them for bonus points!
Answer briefly:
PACMAN: <position>
GHOSTS: <color and position of each ghost>
PELLETS: <which directions have pellets>
DANGER: <which directions have normal ghosts>
GO: <one non-diagonal direction>
```

Each field asks the model to report on a specific part of the image. By the time it reaches `GO:`, it has produced an analysis of the game state — positions, hazards, pellet directions. The direction choice comes *after* that output, not instead of it. The structured prompt changes several things at once — output fields, ghost guidance, `GO:` instead of `MOVE:`, and parser behavior — so the results support that this bundle helped some models, not that any single change caused the gains.

We also added explicit ghost colors (red, pink, orange, cyan) and taught the model about edible blue ghosts — a power pellet mechanic where eating a big pellet makes ghosts vulnerable. The `Exception:` line gives the model a chance to exploit this instead of always running away.

Key design decisions: we switched from `MOVE:` to `GO:` to break the fill-in-the-blank pattern. We said "answer briefly" instead of setting a token limit. We let models say "north" or "upward" and built a flexible parser to handle all synonyms.

### How the Models Actually Reason

Here's what Qwen2.5-VL 3B produces with the structured prompt — concise, field-by-field, and done in 3 seconds:

```
PACMAN: bottom left corner
GHOSTS: Red (top right), Pink (bottom right), Orange (top left), Cyan (bottom left)
PELLETS: Top left and top right
DANGER: Top left, top right
GO: right
```

Contrast that with Pixtral 12B on the free-form prompt — it writes a full analysis, checking each direction in turn:

```
To determine the direction Pac-Man should move, let's analyze the image:
1. **Identify Pac-Man's Current Position**: Pac-Man is in the lower-left corner.
2. **Scan for Pellets and Ghosts in Each Direction**:
   - **Right**: There are pellets, and no ghosts are present.
   - **Left**: There are no pellets, and a ghost is present.
   - **Up**: There are no...
```

Pixtral's approach works well with free-form prompts — it's thorough and varied. But give it the structured template, and it bloats each field into a markdown section with headers and paragraphs, hitting the 150-token limit before ever reaching `GO:`. The structured format that helps smaller models *constrains* Pixtral's natural reasoning style.

### The Tradeoff

Average scores increased for 4 of 7 models in these five-run comparisons — but decreased for 3.

| Model | Free-form Avg (min–max) | Structured Avg (min–max) | Change |
|-------|-------------------------|--------------------------|--------|
| **Qwen2.5-VL 3B** | 411 (240–640) | **516** (280–650) | **+26%** |
| **Qwen3-VL 4B** | 120 (120–120) | **260** (160–330) | **+117%** |
| **Gemma 3 4B** | 120 (120–120) | **248** (230–280) | **+107%** |
| **SmolVLM** | 175 (170–180) | **240** (120–390) | **+37%** |
| Pixtral 12B | 535 (230–1340) | 320 (170–550) | -40% |
| Qwen2.5-VL 7B | 240 (130–470) | 160 (160–160) | -33% |
| Moondream2 | 268 (170–300) | 120 (120–120) | -55% |

Pixtral — the free-form champion — bloats the structured format into markdown reports with headers, hitting the token limit before reaching `GO:`. Moondream2 is the most counterintuitive: with the structured prompt it survives much longer (14→29 turns) but scores *less* (268→120). It's alive longer but completely lost — the template broke its accidental "spam up and collect pellets" strategy without replacing it with real navigation.

**There is no universal prompt.** What fixes broken models constrains good ones. The ideal approach would be per-model prompt selection: free-form for natural reasoners, structured for models that need guidance.

**Overall champion: Qwen2.5-VL 3B on the structured prompt** — 516 avg, beating the random baseline by 66%. A 3-billion-parameter model running on a MacBook. On the free-form prompt, this model also ran at 3.3 seconds per move vs. Pixtral's 14.4 seconds — though latencies may differ across prompt conditions.

One more surprise from the data: **speed doesn't predict quality.** Pixtral 12B takes 14.4 seconds per move. Qwen2.5-VL 3B takes 3.3 seconds. They trade the top spot depending on the prompt version. The fastest model in the benchmark (Qwen3-VL at 2.6 seconds per move) scores the lowest on the free-form prompt. Inference latency didn't predict performance in this benchmark.

---

## Enter Claude Opus

After benchmarking the local models, we wanted to know: what can a frontier model do with the same setup? We played 5 runs with Claude Opus 4.6. Opus received the same raw PNG screenshots, the same structured prompt, and the same 45-frame moves. The difference from local models: Opus runs remotely via API rather than on-device.

| Game | Score | What Happened |
|------|-------|---------------|
| 1 | **930** | Ghost caught at wall-junction stall |
| 2 | 490 | Ghost caught during navigation |
| 3 | 460 | Ghost caught at wall-junction stall |
| 4 | 540 | Ghost caught during navigation |
| 5 | **1010** | Wall stall turns 20–31, broke free, second stall at turn 46 |
| **Avg** | **686** |  |

**686 average (range: 460–1010) — above the best local model (Qwen2.5-VL 3B at 516), and well above the random baseline.** With only 5 runs each and wide score variance, the gap should be read with caution. The more interesting difference is qualitative: Opus appears to reason about the game state differently. With even a small amount of prompt tuning or wall-hit feedback and past moves, it could likely score much higher.

Pixtral and Qwen 3B also reason about the game — they vary directions, analyze each option, and respond to game state. But Opus takes it further: systematic corridor-sweeping, reversing away from ghosts toward uncleared areas, and chasing edible blue ghosts for bonus points. The local models play reactively; Opus plays with something closer to a plan.

[VIDEO: Claude Opus playing Ms. Pac-Man — showing corridor-sweeping strategy and ghost avoidance.]

### The Wall Problem

We saw wall and junction stalls across the tested models — not just Opus. The SNES renders the maze at 256×224 pixels. Corridors and walls are each ~8 pixels wide. After screenshot compression, the difference between "open corridor" and "thin wall" can be 1–2 pixels.

[IMAGE: Screenshot of Pac-Man stuck at a wall junction, with ghosts approaching. Models often struggled to escape corners like this.]

At a T-junction, one direction is open and two are walls — but all three can look alike from a low-res top-down view. Direction spammers hit a wall and stay there. Models that navigate (Qwen 3B, Pixtral, Opus) waste turns trying adjacent directions: 3–5 moves with no score change, and since each move advances 0.75 seconds of in-game time, that's enough for ghosts to close in. In Opus's runs, this accounted for nearly every death.

**Cleared corridors become invisible.** Once pellets are eaten, an open corridor is just black space — visually identical to the space behind walls. The pellets were serving as breadcrumbs. As the maze gets cleared, navigation gets *harder*. This is why every Opus run ended on level 2 — by then, most breadcrumbs are gone.

**No memory between turns.** Each turn is a fresh screenshot. No model can remember which direction it tried last turn. Why not append previous moves as a running log? We tried — most models got confused by the extra context and played worse.

**Ms. Pac-Man's maze design makes it worse.** Unlike original Pac-Man's fixed maze, Ms. Pac-Man rotates between layouts. And the maze is larger than the screen — every model makes local decisions from a partial view.

### Is This a Vision Problem or a Reasoning Problem?

An open question: is the wall failure about *seeing* (visual acuity at 256×224) or *reasoning* (spatial understanding)? A human at the same resolution doesn't get stuck — but humans compensate with memory and feedback. One way to test: preprocess screenshots to explicitly mark open directions and walls, removing perceptual ambiguity entirely.

---

## What We Learned

**In this setup, most small local VLMs couldn't turn screenshots into reliable moves.** The models we tested can describe what they see — colors, shapes, general layout — but struggled to extract the spatial relationships needed for gameplay: what's adjacent to what, which directions are open, where the nearest ghost is. Understanding an image is not the same as acting on it.

**Bigger isn't always better.** The 3B model outperformed its 7B sibling. One possible explanation: the 7B's more polished instruction-following causes it to skip reasoning on short-answer prompts, while the 3B over-explains — which forces it to actually analyze the screenshot.

**Prompt structure matters more than model size.** Qwen2.5-VL 3B on the structured prompt (516) outscored Pixtral 12B on the same prompt (320). Pixtral's best result (535) came with the free-form prompt, where it could reason in its own style. The right prompt for the right model moved the needle more than scaling up parameters.

**Watch your prompt endings.** In our tests, ending a prompt with `MOVE: <direction>` sometimes led models to return a one-token answer without showing evidence of image analysis. Similar prompt endings (like `ANSWER: <value>` or `RESULT: <choice>`) may affect other VLM tasks — it's worth testing the format rather than assuming the model is reasoning through it.

**Local and frontier models hit different ceilings.** Local models were less reliable at turning observations into effective moves. Opus reasons effectively but struggles at wall junctions. Different failure modes at different scales, same outcome: Pac-Man walks into a wall.

**This is a test of single-frame visual decision-making, not general game-playing ability.** With no memory, no feedback, and no context beyond a single screenshot, even the strongest model can't navigate reliably. The setup is deliberately constrained — it isolates how well a model can read and act on a single image.

---

## What's Next

The obvious follow-up: preprocess screenshots to explicitly mark open directions and walls, isolating whether the wall problem is perceptual. Other ideas: wall-hit feedback ("score unchanged — try another direction"), multi-frame context for tracking ghost trajectories, and in-context examples of good play.

If you gave a frontier model like Opus persistent context, wall-hit feedback, and basic game state tracking, we'd expect performance to improve dramatically — its failures appear to be about navigation and memory, not strategy. How far that goes is an open question, but the point of this setup was to test raw single-frame visual decision-making, which is the harder and more revealing question.

But the most interesting takeaway is broader than any specific fix. A simple arcade game turns out to be a surprisingly revealing test of single-frame visual decision-making. It exposes how brittle screenshot-based control can be — prompt sensitivity, direction spam, the gap between describing an image and acting on it. Prompt structure can help, but it doesn't solve perception, memory, or control. And even a frontier model, with all its reasoning power but no memory, hits a perceptual wall (literally) when working from raw low-resolution pixels.

Small vision models have come a long way — they can describe images, answer questions about them, and run on a laptop. But playing a classic arcade game zero-shot, from a single screenshot, with no feedback? That's still a surprisingly hard benchmark.

---

*All experiments ran on an M1 MacBook Pro (16GB) using [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for local models. Claude Opus games used the Anthropic API. The emulator is [Mesen](https://www.mesen.ca/). All code, prompts, Lua scripts, and run data are on [GitHub](https://github.com/grstathis/can-llms-play-pacman).*

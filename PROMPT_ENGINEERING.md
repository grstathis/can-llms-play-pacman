# Prompt Engineering for VLM Game-Playing — The Full Story

## The Problem

We built a pipeline where local vision LLMs play Ms. Pac-Man on a real SNES emulator. The model sees a raw screenshot each turn and must output a direction. Simple enough — except 5 of 7 models just spam a single direction and die.

The journey from "why is everything broken" to "we can fix it with prompt design" turned out to be the most interesting part of the whole project.

---

## Prompt v1: Verbose Chain-of-Thought (Oct 31 2026)

Our first attempt asked the model to reason step by step:

```
Analyze this video game Pac-Man screenshot step by step:
1. Where is the yellow character (Pac-Man)?
2. Ghosts are the small red, pink, orange, or cyan blob-shaped characters
   with eyes. Are there any ghosts in the image?
3. Describe what is directly adjacent to Pac-Man:
   - UP: magenta wall or black corridor or grey/white pellets or ghosts?
   - DOWN: magenta wall or black corridor or grey/white pellets or ghosts?
   - LEFT: magenta wall or black corridor or grey/white pellets or ghosts?
   - RIGHT: magenta wall or black corridor or grey/white pellets or ghosts?

   If it is wall or ghosts it is BLOCKED.

Rules: Aim for the open direction with pellets and avoid blocked paths.

Pick ONE direction as pacman top view controller: up, down, left, or right.
Exception: If you eat a big grey/white pellets then ghosts will turn White
or Dark Blue — chase and eat them for big bonus points!
There is always an open direction, None is not an acceptable answer.
End with: MOVE: <direction>
```

**Result:** ~200 tokens per response. Worked on Qwen2.5-VL 3B but too verbose — slow inference, and smaller models would ramble or lose track of the question mid-response.

---

## Prompt v2: Tight Format (Early Nov 2026)

Compressed to ~50 tokens output:

```
You control Pac-Man (yellow) in this screenshot.
Walls are magenta. Pellets are white/grey dots. Ghosts are red/pink/orange/cyan
blobs with eyes.

For each direction (UP, DOWN, LEFT, RIGHT): wall=BLOCKED, corridor/pellets=OPEN.
Avoid ghosts. Go toward pellets. If ghosts are blue/white, chase them.

MOVE: <up|down|left|right>
```

**Result:** Better latency. Some models followed the format. But the structured UP/DOWN/LEFT/RIGHT analysis was still too much for smaller models — they'd fill in all four and then pick randomly.

---

## Prompt v3: Minimal Fill-in-the-Blank (Nov 6 2026)

Stripped to the bare minimum:

```
This is a Pac-Man game screenshot. Pac-Man is yellow. Walls are magenta and
they block Pac-man. Pellets are white/grey dots. Ghosts are colored blobs
with eyes.
Which direction (left, right, up, down) has pellets and no ghosts?
MOVE: <direction>
```

**Result:** This is where things got interesting — and broke.

### The Disaster

With this prompt, we ran `run_all_evals.sh` across all models. Results:

| Model | Avg Score | Top Direction | % Spam |
|-------|-----------|--------------|--------|
| Pixtral 12B | 535 | down | 36% (varied) |
| Qwen2.5-VL 3B | 411 | right | 34% (varied) |
| Random Baseline | 310 | — | — |
| Moondream2 | 268 | up | **86%** |
| Qwen2.5-VL 7B | 240 | down | **72%** |
| SmolVLM | 175 | right | **87%** |
| Qwen3-VL 4B | 120 | left | **97%** |
| LLaVA 1.6 7B | 28 | — | no valid moves |

**5 of 7 models were direction spammers.** Only Pixtral 12B and Qwen2.5-VL 3B actually played the game. The rest fixated on one direction and repeated it every turn until they died.

### Investigating Qwen3-VL 4B: The Thinking Mode Bug

Qwen3-VL was the most extreme — 97% "left", scoring exactly 120 in all 5 games. But when we asked it to simply *describe* the screenshot, it produced 2000+ chars of detailed, accurate game analysis. The vision encoder worked perfectly.

**Root cause:** Qwen3 models have a hybrid thinking/non-thinking architecture with special `<think>` (token 151667) and `</think>` (token 151668) tokens. The `MOVE: <direction>` template triggered "fill-in-the-blank" behavior — the model completed it as a single token and hit EOS immediately.

We verified this by prepending `<think>\n` to the assistant turn:
- **Without thinking:** `'left'` (4 chars, 1.2s)
- **With thinking:** 641 chars of spatial reasoning, varied directions (29-50s)

The thinking mode was unusable for gameplay (30-50s/turn, looped into "Wait — let me look again" spirals), but it proved the model *could* play if it reasoned first.

### Investigating Qwen2.5-VL 7B: The Bigger-is-Worse Paradox

The 3B and 7B versions of Qwen2.5-VL use the same architecture, same prompt, same pipeline. Side-by-side on the same screenshot:

**Qwen2.5-VL 3B** (works):
```
"To determine the direction with pellets and no ghosts, let's carefully
examine the maze: 1. Top Left Corner: There are no pellets here. 2. Top ..."
(623 chars of reasoning)
```

**Qwen2.5-VL 7B** (broken):
```
"The direction that has pellets and no ghosts is **down**."
(57 chars, no reasoning)
```

The 7B model's instruction-following was *too efficient* — it recognized the short-answer format and skipped reasoning entirely, defaulting to "down" 72% of the time. The 3B's less polished instruction tuning was an advantage: it over-explained, which forced it to actually analyze the screenshot.

### Ruling Out Framework Bugs

Before blaming the prompt, we systematically ruled out other causes:

1. **Official CLI test** (`python -m mlx_vlm.generate`): Same behavior — not our code
2. **mlx-vlm upgrade** (0.7.10 → 0.7.12): No change
3. **HuggingFace cache clear** (deleted model caches, forced re-download): No change
4. **Different image:** Models gave detailed descriptions for non-game prompts

**Conclusion:** The prompt was the problem. Specifically, `MOVE: <direction>` was a trap that short-circuited reasoning in most models.

---

## Prompt v4: "Say Why" (Nov 7 2026)

First fix attempt — force one sentence of reasoning before the answer:

```
You control the yellow Pac-Man in this screenshot. Walls are magenta.
Pellets are white dots. Ghosts are colored blobs with eyes.
Which direction should Pac-Man move to eat pellets while avoiding ghosts?
Say why in one sentence, then: MOVE: direction
```

**Results on the broken models:**

| Model | v3 (broken) | v4 (say why) |
|-------|------------|-------------|
| Qwen3-VL 4B | `'left'` (1 word) | `'Move right to eat pellets...\nMOVE: right'` (17 words) |
| Qwen2.5-VL 7B | `'...is **down**.'` (10 words) | `'...no immediate ghosts. MOVE: right'` (19 words) |
| Qwen2.5-VL 3B | 51 words reasoning | 27 words reasoning (more compact!) |
| Pixtral 12B | works fine | works fine (15 words) |
| SmolVLM | `right` spam | `'Left. Ghosts move faster.'` (6 words) |

**It fixed Qwen3-VL and Qwen2.5-VL 7B!** Both went from single-direction spam to actual reasoning.

However, in the eval run Qwen3-VL 4B still only scored 144 avg (four games at 180, one at 0). The single sentence of reasoning wasn't enough to break it out of its directional bias. We needed something stronger.

---

## Prompt v5: Structured Chain-of-Thought (Nov 7 2026)

The key insight: don't ask the model to reason *then* answer. Ask it explicit questions about the game state that *force* it to look at the screenshot, and constrain answers to be brief:

```
Pac-Man game. Yellow=Pac-Man, magenta=walls, white dots=pellets.
Ghosts: red, pink, orange, cyan blobs with eyes — avoid them.
Exception: blue/white ghosts are EDIBLE — chase them for bonus points!
Answer briefly:
PACMAN: <position>
GHOSTS: <color and position of each ghost>
PELLETS: <which directions have pellets>
DANGER: <which directions have normal ghosts>
GO: <best direction>
```

### Design decisions:

1. **Explicit ghost colors** (red, pink, orange, cyan) — helps the model identify specific game elements instead of guessing
2. **Blue ghost exception** — teaches the power pellet mechanic so the model can exploit it
3. **Structured fields** — each field forces the model to look at a specific aspect of the screenshot
4. **Open-ended GO:** — we don't constrain direction vocabulary. Models can say "north", "top-right", "upward", whatever feels natural. The parser handles all synonyms
5. **"Answer briefly"** — keeps output compact without a rigid token constraint
6. **No `MOVE: <direction>` template** — avoids the fill-in-the-blank trap

### Results across all models (single screenshot test):

| Model | v3 (spam) | v5 (structured) |
|-------|-----------|----------------|
| Qwen3-VL 4B | `'left'` (1w) | Full 5-field response, varied directions across screenshots |
| Qwen2.5-VL 7B | `'...is **down**'` (10w) | Full 5-field response, 17w, varied |
| Qwen2.5-VL 3B | 51w reasoning | 13w structured, concise |
| Pixtral 12B | works | 80w (bloats but works) |
| SmolVLM | `'right'` spam | 11w structured |
| Moondream2 | `'up'` spam | 15w structured |
| Gemma 3 4B | 120 every game | Full structured response, varied |

**Every previously-broken model now produces structured output with varied directions.**

### Direction variance verification:

We tested the previously-broken models on 3 different screenshots:

**Qwen3-VL 4B** (was 97% "left"):
- turn_001: GO: right
- turn_002: GO: right  
- turn_003: GO: up

**Qwen2.5-VL 7B** (was 72% "down"):
- turn_001: GO: up
- turn_002: GO: right
- turn_003: GO: up

Both models now respond to different game states with different directions. The structured questions force them to actually analyze the screenshot before committing to an answer.

### Parser updates:

To support free-form direction vocabulary, we built a comprehensive direction normalizer:

```python
_DIR_SYNONYMS = {
    "up": "up", "north": "up", "top": "up", "upward": "up",
    "down": "down", "south": "down", "bottom": "down", "downward": "down",
    "left": "left", "west": "left",
    "right": "right", "east": "right",
}
```

For compound directions like "south-east" or "top-left", the parser takes the first component (vertical wins over horizontal). The `GO:` field is parsed first with highest priority, falling back to `MOVE:`, then free-text direction words.

---

## The Lesson

The same vision model can appear broken or functional depending entirely on prompt structure:

**Fill-in-the-blank prompts** (`MOVE: <direction>`) are a trap for instruction-tuned models. They trigger pattern completion — the model fills in one token and stops. Larger, better-instruction-tuned models are *more* susceptible because they're better at recognizing short-answer patterns.

**Structured chain-of-thought prompts** force the model to engage with the image before answering. By asking explicit questions (where is Pac-Man? where are the ghosts?), you guarantee the model processes the visual input. The direction answer comes *after* the model has already committed to a spatial understanding.

**But structure has a cost.** Models that already reason naturally (Pixtral 12B, Qwen2.5-VL 3B) perform worse when forced into a rigid template. Pixtral bloats the structured format into markdown reports and hits the token limit before reaching the answer. The 3B parrots the template back instead of filling it in. The structure that saves broken models constrains good ones.

**Let models speak their own language.** Constraining output to "up/down/left/right" biases models toward memorized patterns. Letting them say "north", "top-left", or "upward" and parsing flexibly leads to more genuine reasoning.

**There is no universal prompt.** The ideal approach is per-model prompt selection — free-form for models that naturally reason, structured for models that need guidance.

---

## v5 Eval Results

Full eval run: 7 models × 5 games each (some with Mesen boot failures), all using v5 structured prompt with ghost colors and non-diagonal direction constraint.

| Model | v3 Avg | v5 Avg | v5 Games | Change | Verdict |
|-------|--------|--------|----------|--------|---------|
| **Qwen3-VL 4B** | 120 | **260** | 5 | **+117%** | Fixed — varied directions, real gameplay |
| **Gemma 3 4B** | 120 | **248** | 5 | **+107%** | Fixed — consistent 230-280 range |
| **SmolVLM** | 175 | **240** | 3 | **+37%** | Improved — broke out of "right" spam |
| Pixtral 12B | 535 | **320** | 6 | **-40%** | Hurt — bloats to markdown, truncated |
| Qwen2.5-VL 7B | 240 | 160 | 1 | -33% | Worse — hallucinated grid coordinates |
| Moondream2 | 268 | 120 | 2 | -55% | Broken — parroted template format |
| Qwen2.5-VL 3B | 411 | **516** | 5 | **+26%** | New champion — structured format focuses natural reasoning |

Note: Qwen2.5-VL 3B initially appeared to crash with v5 (echoing the template back). This was caused by Mesen emulator boot failures, not a prompt issue. A clean rerun produced the best scores in the entire benchmark (650, 440, 580, 630, 280).

### Best performers by prompt version:

**v3 winners** (free-form reasoning): Pixtral 12B (535), Qwen2.5-VL 3B (411)
**v5 winners** (structured CoT): **Qwen2.5-VL 3B (516)**, Pixtral 12B (320), Qwen3-VL 4B (260), Gemma 3 4B (248), SmolVLM (240)

**Overall champion: Qwen2.5-VL 3B on v5** — 516 avg, 3.3s/move, beats random baseline by 66%.

### The prompt-model interaction matrix:

Models that **benefit from structure** (v5 > v3):
- Qwen2.5-VL 3B — already reasoning naturally, structure focuses it (+26%)
- Models that short-circuit on `MOVE: <direction>` (Qwen3-VL thinking mode bug, Gemma flat-lining)
- Models that spam one direction without reasoning (SmolVLM)

Models that **need freedom** (v3 > v5):
- Verbose models that bloat structured formats (Pixtral 12B — bloats to markdown, truncated)
- Qwen2.5-VL 7B — produces hallucinated grid coordinates instead of real positions
- Very small models confused by templates (Moondream2)

---

## Timeline

| Date | Prompt | Key Event |
|------|--------|-----------|
| Oct 31 | v1 (verbose CoT) | First working prototype with Qwen2.5-VL 3B |
| Early Nov | v2 (tight format) | Compressed for speed |
| Nov 6 | v3 (minimal) | Full eval run — discovered 5/7 models spam |
| Nov 7 00:39 | v3 finalized | brain.py updated, run_all_evals.sh batch |
| Nov 7 AM | Investigation | Direction analysis reveals spam patterns |
| Nov 7 PM | Qwen3-VL deep-dive | Thinking mode bug found, CLI verification |
| Nov 7 PM | 3B vs 7B comparison | Bigger model worse due to efficient instruction-following |
| Nov 7 PM | v4 ("say why") | Fixes Qwen models in tests but Qwen3-VL still weak in eval (144 avg) |
| Nov 7 PM | v5 (structured CoT) | All 7 models produce varied, structured output in tests |
| Nov 7 PM | v5 + ghost colors | Added explicit ghost colors and blue ghost exception |
| Nov 7 Eve | v5 eval complete | Fixes 4 models, hurts 3; Qwen2.5-VL 3B crash was Mesen bug |
| Nov 7 Late | v5 3B rerun | Qwen2.5-VL 3B hits 516 avg — new overall champion |

---

## Prompt Versions Quick Reference

| Version | Strategy | Tokens Out | Models Helped | Models Hurt | Key Insight |
|---------|----------|-----------|--------------|------------|-------------|
| v1 | Verbose step-by-step | ~200 | Qwen2.5-VL 3B | Most (too slow) | Reasoning helps but costs too much |
| v2 | Per-direction analysis | ~50 | 2-3 | Small models | Still too structured for small models |
| v3 | Fill-in-the-blank | ~5-50 | Pixtral, 3B | 5 others (spam) | `MOVE:` template is a single-token trap |
| v4 | "Say why" + MOVE | ~15-30 | Qwen3, 7B | None tested | Not enough reasoning for eval performance |
| v5 | Structured fields + GO | ~15-80 | 3B, Qwen3, Gemma, SmolVLM | Pixtral, 7B, Moondream2 | Best overall; 3B rerun hit 516 avg |

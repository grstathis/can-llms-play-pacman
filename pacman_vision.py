"""OpenCV preprocessor for Ms. Pac-Man screenshots.

Detects Pac-Man, ghosts, dots, and walls using color masks.
Produces an annotated image and a game state dict to help small VLMs play better.

Usage:
    python pacman_vision.py screen.png                  # show annotated image
    python pacman_vision.py screen.png --save out.png   # save annotated image
    python pacman_vision.py screen.png --json            # print game state as JSON

As a module:
    from pacman_vision import analyze_screenshot
    state, annotated_img = analyze_screenshot("screen.png")
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# HSV ranges measured from actual SNES Ms. Pac-Man screenshots
# OpenCV HSV: H 0-179, S 0-255, V 0-255

# Pac-Man: bright saturated yellow (H 15-28, S 230+)
# Orange ghost: duller orange-yellow (H 28-40, S 100-200)
PACMAN_HSV = (np.array([15, 220, 180]), np.array([28, 255, 255]))
RED_HSV = ((np.array([0, 180, 120]), np.array([6, 255, 255])),
           (np.array([174, 180, 120]), np.array([179, 255, 255])))
CYAN_HSV = (np.array([80, 150, 150]), np.array([105, 255, 255]))
ORANGE_HSV = (np.array([8, 180, 150]), np.array([18, 255, 255]))
SCARED_HSV = (np.array([105, 150, 150]), np.array([125, 255, 255]))
WALL_HSV = (np.array([140, 25, 100]), np.array([175, 255, 255]))
DOT_HSV = (np.array([0, 0, 110]), np.array([180, 70, 255]))

MIN_GHOST_AREA = 20
MAX_GHOST_AREA = 300
MIN_DOT_AREA = 1
MAX_DOT_AREA = 25
MIN_PACMAN_AREA = 30

SCAN_DIST = 30


def _find_blobs(mask, min_area, max_area):
    """Find blob centers and areas from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                blobs.append((cx, cy, area))
    return blobs


def _largest_blob(mask, min_area=10):
    """Return center of the largest blob, or None."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > best_area and area >= min_area:
            best_area = area
            m = cv2.moments(c)
            if m["m00"] > 0:
                best = (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]), area)
    return best


def _direction_from(src, dst):
    """Cardinal direction from src to dst."""
    dx = dst[0] - src[0]
    dy = dst[1] - src[1]
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


def _distance(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _check_wall(wall_mask, px, py, direction):
    """Check if movement in this direction is blocked.

    Scans along the center line. If any row within SCAN_DIST has
    3+ out of 5 pixels as wall, the path is blocked.
    """
    h, w = wall_mask.shape
    for d in range(4, SCAN_DIST + 1):
        wall_count = 0
        for offset in range(-2, 3):
            if direction == "up":
                ny, nx = py - d, px + offset
            elif direction == "down":
                ny, nx = py + d, px + offset
            elif direction == "left":
                ny, nx = py + offset, px - d
            elif direction == "right":
                ny, nx = py + offset, px + d
            else:
                continue
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                wall_count += 1
            elif wall_mask[ny, nx] > 0:
                wall_count += 1
        if wall_count >= 3:
            return True
    return False


def _count_dots_direction(dot_positions, px, py, direction):
    """Count dots in a direction from Pac-Man within a cone."""
    count = 0
    for dx, dy in dot_positions:
        rx, ry = dx - px, dy - py
        dist = (rx * rx + ry * ry) ** 0.5
        if dist < 5 or dist > 100:
            continue
        if direction == "up" and ry < -5 and abs(rx) < abs(ry):
            count += 1
        elif direction == "down" and ry > 5 and abs(rx) < abs(ry):
            count += 1
        elif direction == "left" and rx < -5 and abs(ry) < abs(rx):
            count += 1
        elif direction == "right" and rx > 5 and abs(ry) < abs(rx):
            count += 1
    return count


def analyze_screenshot(image_path: str | Path, save_path: str | Path | None = None):
    """Analyze a Ms. Pac-Man screenshot.

    Returns (state_dict, annotated_image).
    If save_path is given, writes the annotated image to disk.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    annotated = img.copy()
    h, w = img.shape[:2]

    # 1. Find ghosts FIRST (so we can exclude orange ghost from Pac-Man)
    ghosts = []

    # Red ghost
    red_mask = cv2.inRange(hsv, *RED_HSV[0]) | cv2.inRange(hsv, *RED_HSV[1])
    for cx, cy, area in _find_blobs(red_mask, MIN_GHOST_AREA, MAX_GHOST_AREA):
        ghosts.append({"pos": (cx, cy), "color": "red", "scared": False})

    # Cyan ghost
    cyan_mask = cv2.inRange(hsv, *CYAN_HSV)
    for cx, cy, area in _find_blobs(cyan_mask, MIN_GHOST_AREA, MAX_GHOST_AREA):
        ghosts.append({"pos": (cx, cy), "color": "cyan", "scared": False})

    # Orange ghost — same yellow as Pac-Man, but has red pixels (eyes) nearby.
    # Find yellow blobs that have red pixels within a small radius.
    yellow_mask = cv2.inRange(hsv, np.array([15, 180, 180]), np.array([35, 255, 255]))
    yellow_mask[:10, :] = 0
    yellow_mask[-25:, :] = 0
    red_nearby = cv2.dilate(red_mask, np.ones((6, 6)))
    orange_candidates = _find_blobs(yellow_mask, MIN_GHOST_AREA, MAX_GHOST_AREA)
    orange_positions = []
    for cx, cy, area in orange_candidates:
        if red_nearby[cy, cx] > 0:
            ghosts.append({"pos": (cx, cy), "color": "orange", "scared": False})
            orange_positions.append((cx, cy))

    # Scared (blue) ghosts
    scared_mask = cv2.inRange(hsv, *SCARED_HSV)
    for cx, cy, area in _find_blobs(scared_mask, MIN_GHOST_AREA, MAX_GHOST_AREA):
        ghosts.append({"pos": (cx, cy), "color": "blue", "scared": True})

    # 2. Find Pac-Man — exclude known orange ghost positions
    pac_mask = cv2.inRange(hsv, np.array([15, 200, 180]), np.array([30, 255, 255]))
    pac_mask[:10, :] = 0
    pac_mask[-25:, :] = 0
    for ox, oy in orange_positions:
        cv2.circle(pac_mask, (ox, oy), 15, 0, -1)
    pac = _largest_blob(pac_mask, 10)
    pac_pos = (pac[0], pac[1]) if pac else None

    # Scared (blue) ghosts
    scared_mask = cv2.inRange(hsv, *SCARED_HSV)
    for cx, cy, area in _find_blobs(scared_mask, MIN_GHOST_AREA, MAX_GHOST_AREA):
        ghosts.append({"pos": (cx, cy), "color": "blue", "scared": True})

    # Add direction from Pac-Man
    for g in ghosts:
        if pac_pos:
            g["dir_from_pac"] = _direction_from(pac_pos, g["pos"])
            g["distance"] = round(_distance(pac_pos, g["pos"]))
        else:
            g["dir_from_pac"] = "unknown"
            g["distance"] = 999

    # 3. Find dots and power pellets
    dot_mask = cv2.inRange(hsv, *DOT_HSV)
    wall_mask = cv2.inRange(hsv, *WALL_HSV)
    dot_mask = dot_mask & ~wall_mask
    dot_blobs = _find_blobs(dot_mask, MIN_DOT_AREA, 200)

    regular_dots = dot_blobs
    dot_positions = [(cx, cy) for cx, cy, _ in regular_dots]

    # Count dots per direction from Pac-Man
    dots_per_dir = {}
    if pac_pos:
        for d in ("up", "down", "left", "right"):
            dots_per_dir[d] = _count_dots_direction(dot_positions, pac_pos[0], pac_pos[1], d)
    else:
        dots_per_dir = {"up": 0, "down": 0, "left": 0, "right": 0}

    # 4. Check walls adjacent to Pac-Man
    walls = {}
    if pac_pos:
        for d in ("up", "down", "left", "right"):
            walls[d] = _check_wall(wall_mask, pac_pos[0], pac_pos[1], d)
    else:
        walls = {"up": False, "down": False, "left": False, "right": False}

    # 5. Find nearest ghost and nearest dot direction
    nearest_ghost = None
    nearest_ghost_dist = 999
    for g in ghosts:
        if not g["scared"] and g["distance"] < nearest_ghost_dist:
            nearest_ghost = g
            nearest_ghost_dist = g["distance"]

    best_dot_dir = max(dots_per_dir, key=dots_per_dir.get) if dots_per_dir else "right"

    # Build state dict
    state = {
        "pacman": pac_pos,
        "ghosts": ghosts,
        "dots_up": dots_per_dir.get("up", 0),
        "dots_down": dots_per_dir.get("down", 0),
        "dots_left": dots_per_dir.get("left", 0),
        "dots_right": dots_per_dir.get("right", 0),
        "total_dots": len(regular_dots),
        "wall_up": walls.get("up", False),
        "wall_down": walls.get("down", False),
        "wall_left": walls.get("left", False),
        "wall_right": walls.get("right", False),
        "nearest_dot_dir": best_dot_dir,
        "nearest_ghost_dir": nearest_ghost["dir_from_pac"] if nearest_ghost else "none",
        "nearest_ghost_dist": nearest_ghost_dist,
    }

    # --- Draw annotations ---

    # Draw regular dots (white)
    for cx, cy, area in regular_dots:
        cv2.circle(annotated, (cx, cy), 3, (220, 220, 220), -1)

    # Draw Pac-Man
    if pac_pos:
        cv2.circle(annotated, pac_pos, 12, (0, 255, 0), 2)
        cv2.putText(annotated, "PAC", (pac_pos[0] - 12, pac_pos[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Draw ghosts
    for g in ghosts:
        gx, gy = g["pos"]
        if g["scared"]:
            color = (255, 200, 0)
            label = "EAT!"
        else:
            color = (0, 0, 255)
            label = f"GHOST"
        cv2.circle(annotated, (gx, gy), 10, color, 2)
        cv2.putText(annotated, label, (gx - 15, gy - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        if pac_pos and not g["scared"]:
            cv2.arrowedLine(annotated, pac_pos, (gx, gy), (0, 0, 200), 1, tipLength=0.15)

    # Draw direction indicators from Pac-Man
    if pac_pos:
        px, py = pac_pos
        arrow_len = 25
        for d, (dx, dy) in [("up", (0, -1)), ("down", (0, 1)),
                             ("left", (-1, 0)), ("right", (1, 0))]:
            end = (px + dx * arrow_len, py + dy * arrow_len)
            if walls[d]:
                cv2.line(annotated, (px + dx * 8, py + dy * 8), end, (0, 0, 255), 2)
                cv2.putText(annotated, "X", (end[0] - 4, end[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            else:
                dot_count = dots_per_dir[d]
                arrow_color = (0, 255, 0) if dot_count > 3 else (0, 200, 200)
                cv2.arrowedLine(annotated, (px + dx * 8, py + dy * 8), end,
                                arrow_color, 2, tipLength=0.3)
                cv2.putText(annotated, str(dot_count), (end[0] - 4, end[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, arrow_color, 1)

    # --- Build clean schematic (black background, only markers) ---
    schematic = np.zeros_like(img)

    # Draw walls as dark grey outlines for spatial context
    schematic[wall_mask > 0] = (40, 40, 40)

    # Draw regular dots as white
    for cx, cy, area in regular_dots:
        cv2.circle(schematic, (cx, cy), 3, (220, 220, 220), -1)

    # Draw Pac-Man as bright green with label
    if pac_pos:
        cv2.circle(schematic, pac_pos, 8, (0, 255, 0), -1)
        cv2.putText(schematic, "YOU", (pac_pos[0] - 12, pac_pos[1] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Draw direction arrows
        px, py = pac_pos
        for d, (dx, dy) in [("up", (0, -1)), ("down", (0, 1)),
                             ("left", (-1, 0)), ("right", (1, 0))]:
            end = (px + dx * 22, py + dy * 22)
            if walls[d]:
                cv2.line(schematic, (px + dx * 10, py + dy * 10), end, (0, 0, 180), 2)
                cv2.putText(schematic, "X", (end[0] - 4, end[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
            else:
                cnt = dots_per_dir[d]
                color = (0, 255, 0) if cnt > 3 else (0, 180, 180)
                cv2.arrowedLine(schematic, (px + dx * 10, py + dy * 10),
                                end, color, 2, tipLength=0.3)
                cv2.putText(schematic, str(cnt), (end[0] - 4, end[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    # Draw ghosts as red/blue
    for g in ghosts:
        gx, gy = g["pos"]
        if g["scared"]:
            cv2.circle(schematic, (gx, gy), 8, (255, 150, 0), -1)
            cv2.putText(schematic, "EAT", (gx - 10, gy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 0), 1)
        else:
            cv2.circle(schematic, (gx, gy), 8, (0, 0, 255), -1)
            cv2.putText(schematic, "G", (gx - 4, gy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(str(save_path), annotated)

    return state, annotated, schematic


def print_state(state: dict):
    """Print a human-readable game state summary."""
    if not state:
        print("No state detected.")
        return

    pac = state["pacman"]
    print(f"Pac-Man: {'(' + str(pac[0]) + ', ' + str(pac[1]) + ')' if pac else 'NOT FOUND'}")
    print(f"Dots: up={state['dots_up']} down={state['dots_down']} "
          f"left={state['dots_left']} right={state['dots_right']} (total: {state['total_dots']})")
    print(f"Walls: up={'WALL' if state['wall_up'] else 'open'} "
          f"down={'WALL' if state['wall_down'] else 'open'} "
          f"left={'WALL' if state['wall_left'] else 'open'} "
          f"right={'WALL' if state['wall_right'] else 'open'}")
    print(f"Best dot direction: {state['nearest_dot_dir']}")

    if state["ghosts"]:
        print(f"Ghosts ({len(state['ghosts'])}):")
        for g in state["ghosts"]:
            scared = " [SCARED - EAT!]" if g["scared"] else ""
            print(f"  {g['color']}: {g['dir_from_pac']} ({g['distance']}px){scared}")
    else:
        print("Ghosts: none visible")

    if state["nearest_ghost_dir"] != "none":
        print(f"DANGER: nearest ghost {state['nearest_ghost_dir']} at {state['nearest_ghost_dist']}px")


def main():
    parser = argparse.ArgumentParser(description="Ms. Pac-Man Screenshot Analyzer")
    parser.add_argument("image", help="Path to screenshot PNG")
    parser.add_argument("--save", help="Save annotated image to this path")
    parser.add_argument("--json", action="store_true", help="Output game state as JSON")
    args = parser.parse_args()

    if not Path(args.image).exists():
        sys.exit(f"Image not found: {args.image}")

    save_path = args.save or (str(Path(args.image).stem) + "_annotated.png")
    state, annotated, schematic = analyze_screenshot(args.image, save_path)

    if state is None:
        sys.exit("Failed to load image")

    schematic_path = str(Path(args.image).stem) + "_schematic.png"
    cv2.imwrite(schematic_path, schematic)

    if args.json:
        print(json.dumps(state, indent=2))
    else:
        print_state(state)
        print(f"\nAnnotated image saved: {save_path}")


if __name__ == "__main__":
    main()

import os
import json
import random
import math
import numpy as np
from PIL import Image, ImageDraw
from datetime import datetime, timezone

# --- CONFIGURATION ---
GRID_SIZE = 15
CELL_SIZE = 32
PREP_STEPS = 15
TOTAL_STEPS = 60

# Tile IDs
EMPTY = 0
WALL = 1
BOX = 2

# Fallback Colors (RGB / RGBA)
COLOR_BG = (30, 30, 36)
COLOR_GRID = (45, 45, 55)
COLOR_WALL = (100, 100, 115)
COLOR_BOX = (210, 150, 75)
COLOR_HIDER = (65, 165, 245)
COLOR_SEEKER = (240, 80, 80)
COLOR_VISION = (240, 80, 80, 45)


def load_sprite(filename):
    path = os.path.join("assets", filename)
    if os.path.exists(path):
        return (
            Image.open(path)
            .convert("RGBA")
            .resize((CELL_SIZE, CELL_SIZE), Image.Resampling.NEAREST)
        )
    return None


SPRITES = {
    "floor": load_sprite("floor.png"),
    "wall": load_sprite("wall.png"),
    "box": load_sprite("box.png"),
    "hider": load_sprite("hider.png"),
    "seeker": load_sprite("seeker.png"),
}


class HideAndSeekEnv:
    def __init__(self):
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        self._build_map()
        # Spawn positions [row, col]
        self.hiders = [[2, 2], [4, 3]]
        self.seekers = [[12, 12], [11, 13]]
        self.caught = [False, False]

    def _build_map(self):
        # Outer borders
        self.grid[0, :] = WALL
        self.grid[-1, :] = WALL
        self.grid[:, 0] = WALL
        self.grid[:, -1] = WALL

        # Room walls with doorways
        self.grid[6, 1:5] = WALL
        self.grid[6, 6:10] = WALL
        self.grid[1:5, 9] = WALL

        # Movable boxes near doorways
        self.grid[5, 5] = BOX
        self.grid[5, 8] = BOX
        self.grid[7, 5] = BOX
        self.grid[4, 8] = BOX

    def has_line_of_sight(self, pos1, pos2):
        r0, c0 = pos1
        r1, c1 = pos2
        dist = math.hypot(r1 - r0, c1 - c0)
        if dist > 7.5:  # Max vision range
            return False
        steps = int(dist * 2)
        if steps == 0:
            return True
        for i in range(1, steps):
            t = i / steps
            r = int(round(r0 + (r1 - r0) * t))
            c = int(round(c0 + (c1 - c0) * t))
            if self.grid[r, c] in (WALL, BOX):
                return False
        return True

    def move_agent(self, pos, dr, dc, can_push=False):
        nr, nc = pos[0] + dr, pos[1] + dc
        if not (0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE):
            return pos

        # Prevent agents from overlapping on the same tile
        if [nr, nc] in self.hiders or [nr, nc] in self.seekers:
            return pos

        target = self.grid[nr, nc]
        if target == EMPTY:
            return [nr, nc]
        elif target == BOX and can_push:
            nnr, nnc = nr + dr, nc + dc
            if (
                0 <= nnr < GRID_SIZE
                and 0 <= nnc < GRID_SIZE
                and self.grid[nnr, nnc] == EMPTY
            ):
                if [nnr, nnc] not in self.hiders and [nnr, nnc] not in self.seekers:
                    self.grid[nr, nc] = EMPTY
                    self.grid[nnr, nnc] = BOX
                    return [nr, nc]
        return pos

    def step(self, current_step):
        is_prep = current_step < PREP_STEPS
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 1. Hiders AI (Push boxes during prep, flee from seekers later)
        for i, h_pos in enumerate(self.hiders):
            if self.caught[i]:
                continue
            best_move = (0, 0)
            if is_prep:
                # Move randomly with bias to push nearby boxes
                random.shuffle(dirs)
                for dr, dc in dirs:
                    nr, nc = h_pos[0] + dr, h_pos[1] + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if self.grid[nr, nc] == BOX:
                            best_move = (dr, dc)
                            break
                if best_move == (0, 0):
                    best_move = random.choice(dirs)
            else:
                # Maximize distance from seekers
                best_dist = -1
                for dr, dc in dirs + [(0, 0)]:
                    nr, nc = h_pos[0] + dr, h_pos[1] + dc
                    if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                        if self.grid[nr, nc] == EMPTY or (dr == 0 and dc == 0):
                            d = min(
                                math.hypot(nr - s[0], nc - s[1]) for s in self.seekers
                            )
                            if d > best_dist:
                                best_dist = d
                                best_move = (dr, dc)

            self.hiders[i] = self.move_agent(
                h_pos, best_move[0], best_move[1], can_push=is_prep
            )

        # 2. Seekers AI (Frozen during prep, chase visible hiders later)
        if not is_prep:
            for i, s_pos in enumerate(self.seekers):
                visible_hiders = [
                    h
                    for idx, h in enumerate(self.hiders)
                    if not self.caught[idx] and self.has_line_of_sight(s_pos, h)
                ]
                best_move = random.choice(dirs)
                if visible_hiders:
                    # Target closest visible hider
                    target = min(
                        visible_hiders,
                        key=lambda h: math.hypot(h[0] - s_pos[0], h[1] - s_pos[1]),
                    )
                    best_dist = 999
                    for dr, dc in dirs:
                        nr, nc = s_pos[0] + dr, s_pos[1] + dc
                        if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                            if self.grid[nr, nc] == EMPTY:
                                d = math.hypot(nr - target[0], nc - target[1])
                                if d < best_dist:
                                    best_dist = d
                                    best_move = (dr, dc)
                self.seekers[i] = self.move_agent(
                    s_pos, best_move[0], best_move[1], can_push=False
                )

        # 3. Check tags
        for idx, h in enumerate(self.hiders):
            for s in self.seekers:
                if math.hypot(h[0] - s[0], h[1] - s[1]) <= 1.1:
                    self.caught[idx] = True

    def render_frame(self, current_step):
        img_size = GRID_SIZE * CELL_SIZE
        img = Image.new("RGBA", (img_size, img_size), COLOR_BG)
        draw = ImageDraw.Draw(img)

        # 1. Draw floor, walls, and boxes
        for r in range(GRID_SIZE):
            for c in range(GRID_SIZE):
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                x1, y1 = x0 + CELL_SIZE, y0 + CELL_SIZE

                if SPRITES["floor"]:
                    img.paste(SPRITES["floor"], (x0, y0))
                else:
                    draw.rectangle([x0, y0, x1, y1], outline=COLOR_GRID)

                tile = self.grid[r, c]
                if tile == WALL:
                    if SPRITES["wall"]:
                        img.paste(SPRITES["wall"], (x0, y0), SPRITES["wall"])
                    else:
                        draw.rectangle([x0 + 1, y0 + 1, x1 - 1, y1 - 1], fill=COLOR_WALL)
                elif tile == BOX:
                    if SPRITES["box"]:
                        img.paste(SPRITES["box"], (x0, y0), SPRITES["box"])
                    else:
                        draw.rectangle([x0 + 4, y0 + 4, x1 - 4, y1 - 4], fill=COLOR_BOX)

        # 2. Draw Seeker Vision Cone (after prep phase)
        if current_step >= PREP_STEPS:
            overlay = Image.new("RGBA", (img_size, img_size), (0, 0, 0, 0))
            ov_draw = ImageDraw.Draw(overlay)
            for s in self.seekers:
                for r in range(GRID_SIZE):
                    for c in range(GRID_SIZE):
                        if self.has_line_of_sight(s, [r, c]):
                            x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                            ov_draw.rectangle(
                                [x0, y0, x0 + CELL_SIZE, y0 + CELL_SIZE],
                                fill=COLOR_VISION,
                            )
            img = Image.alpha_composite(img, overlay)
            draw = ImageDraw.Draw(img)

        # 3. Draw Hiders
        for idx, (r, c) in enumerate(self.hiders):
            if not self.caught[idx]:
                x0, y0 = c * CELL_SIZE, r * CELL_SIZE
                if SPRITES["hider"]:
                    img.paste(SPRITES["hider"], (x0, y0), SPRITES["hider"])
                else:
                    draw.ellipse(
                        [x0 + 5, y0 + 5, x0 + CELL_SIZE - 5, y0 + CELL_SIZE - 5],
                        fill=COLOR_HIDER,
                    )

        # 4. Draw Seekers
        for r, c in self.seekers:
            x0, y0 = c * CELL_SIZE, r * CELL_SIZE
            if SPRITES["seeker"]:
                img.paste(SPRITES["seeker"], (x0, y0), SPRITES["seeker"])
            else:
                draw.ellipse(
                    [x0 + 4, y0 + 4, x0 + CELL_SIZE - 4, y0 + CELL_SIZE - 4],
                    fill=COLOR_SEEKER,
                )

        return img.convert("RGB")


def main():
    if os.path.exists("state.json"):
        with open("state.json", "r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {"match_id": 0, "hider_wins": 0, "seeker_wins": 0}

    env = HideAndSeekEnv()
    frames = []

    for step in range(TOTAL_STEPS):
        env.step(step)
        frames.append(env.render_frame(step))
        if all(env.caught):
            break

    # Save animated GIF
    frames[0].save(
        "match.gif",
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
    )

    # Update stats
    state["match_id"] += 1
    if all(env.caught):
        winner = "🔴 Seekers"
        state["seeker_wins"] += 1
    else:
        winner = "🔵 Hiders"
        state["hider_wins"] += 1

with open("state.json", "w", encoding="utf-8") as f:
json.dump(state, f, indent=2)

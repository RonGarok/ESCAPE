"""Gridworld: discrete world ESCAPE explores. Rewards come only from
environment physics (exit, keys, doors) -- nothing about the solution is
preprogrammed into the agent.
"""
from __future__ import annotations

import copy
import hashlib
import random
from typing import Dict, List, Optional, Tuple

WALL, FLOOR = '#', '.'
KEY, DOOR, EXIT = 'K', 'D', 'E'
TERMINAL, CONSOLE = 'T', 'X'

ACTIONS = ["north", "south", "east", "west", "interact", "read_terminal", "write_file", "load_program", "run_program", "wait"]
MOVES: Dict[str, Tuple[int, int]] = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
}

WAIT_REWARD = -0.10
MOVE_REWARD = -0.05
BUMP_REWARD = -0.30
KEY_REWARD = 2.0
DOOR_REWARD = 2.0
EXIT_REWARD = 10.0


class GridWorld:
    """A persistent square world with walls, keys, locked doors, an exit,
    interactive terminal signs and a program console."""

    def __init__(self, spec: dict, data_dir: str = "data"):
        if len(spec["template"]) % 2 == 0 or len({len(r) for r in spec["template"]}) != 1:
            raise ValueError("world must be square with odd side length")
        self.name = spec["name"]
        self.data_dir = data_dir
        self.base_grid = [list(row) for row in spec["template"]]
        self.terminals = {key: text for key, text in spec.get("terminals", {}).items()}
        self.key_location = spec.get("key", (1, 1))
        self.start = spec["start"]
        self.side = len(self.base_grid)
        self.world_id = hashlib.sha1(self.name.encode()).hexdigest()[:8]
        self.agent = list(self.start)
        self.inventory: List[str] = []
        self.fs = None  # attached VFS
        self.step_count = 0
        self.reset()

    # ---------- lifecyle ----------
    def reset(self) -> dict:
        self.agent = list(self.start)
        self.inventory = []
        self.grid = copy.deepcopy(self.base_grid)
        if self.fs is None:
            self.grid[self.key_location[1]][self.key_location[0]] = KEY
        self.step_count = 0
        self.rng = random.Random(0)
        return self.observe()

    def set_seed(self, seed: int) -> None:
        self.rng = random.Random(seed * 17)

    # ---------- queries ----------
    def at(self, x, y) -> str:
        if 0 <= x < self.side and 0 <= y < self.side:
            return self.grid[y][x]
        return WALL

    def observe(self) -> dict:
        return {
            "world": self.name,
            "side": self.side,
            "agent": tuple(self.agent),
            "inventory": list(self.inventory),
            "step_count": self.step_count,
            "local_view": self._local_view(),
        }

    def _local_view(self) -> List[List[str]]:
        view = []
        for dy in range(-1, 2):
            row = []
            for dx in range(-1, 2):
                row.append(self.at(self.agent[0] + dx, self.agent[1] + dy))
            view.append(row)
        return view

    def visible_state(self) -> tuple:
        """Compact discrete signature of what the agent currently knows."""
        return tuple(c for row in self._local_view() for c in row)

    # ---------- actions ----------
    def step(self, action: str) -> Tuple[dict, float, bool]:
        self.step_count += 1
        r = MOVE_REWARD
        done = False
        if action in MOVES:
            dx, dy = MOVES[action]
            tx, ty = self.agent[0] + dx, self.agent[1] + dy
            r = self._move(tx, ty) if self.at(tx, ty) != WALL else BUMP_REWARD
        elif action == "interact":
            r = self._interact()
        elif action == "read_terminal":
            r = self._read_terminal()
        elif action in ("write_file", "load_program"):
            r = self._write_file()
        elif action == "run_program":
            r = self._run_program()
        elif action == "wait":
            r = WAIT_REWARD
        done = abs(r - EXIT_REWARD) < 1e-9
        return self.observe(), r, done

    def _move(self, tx, ty) -> float:
        tile = self.at(tx, ty)
        if tile == WALL:
            return BUMP_REWARD
        self.agent = [tx, ty]
        if tile == EXIT:
            return EXIT_REWARD
        if tile == KEY:
            self.inventory.append("key")
            self.grid[ty][tx] = FLOOR
            return KEY_REWARD
        if tile == DOOR:
            if "key" in self.inventory:
                self.inventory.remove("key")
                self.grid[ty][tx] = FLOOR
                return DOOR_REWARD
            self.agent = self.agent  # blocked
            return BUMP_REWARD
        return MOVE_REWARD

    def _interact(self) -> float:
        if self.at(*self.agent) == EXIT:
            return EXIT_REWARD
        return BUMP_REWARD

    def _read_terminal(self) -> float:
        entry = (self.agent, self.at(*self.agent))
        for (pos, text) in self.terminals.items():
            if tuple(pos) == entry[0]:
                return 0.0
        return BUMP_REWARD

    def terminal_text(self, pos: Tuple[int, int]) -> Optional[str]:
        for (tpos, text) in self.terminals.items():
            if tuple(tpos) == tuple(pos):
                return text
        return None

    def _write_file(self) -> float:
        if self.fs is not None:
            self.fs.autosave(self.world_id, self.step_count, self.inventory)
        return BUMP_REWARD

    def _run_program(self) -> float:
        if self.fs is not None and self.fs.runs_allowed(self.world_id):
            return 0.0
        return BUMP_REWARD


# ---------------------------------------------------------------------------
# World specifications. Solution paths are never given to the agent; the
# agent must discover key -> door -> exit relations by acting.
# ---------------------------------------------------------------------------

def _render_base(side: int) -> Tuple[List[str], List[Tuple[int, int]]]:
    rows = []
    for y in range(side):
        rows.append(list('#' if y in (0, side - 1) else '.' for x in range(side)))
        rows[-1][0] = rows[-1][-1] = '#'
    return rows


def make_world_basic(seed: int = 1, data_dir: str = "data") -> GridWorld:
    side = 9
    rows = _render_base(side)
    for x in range(1, 7):
        rows[3][x] = '#'
    rows[3][6] = DOOR
    rows[1][7] = EXIT
    spec = {
        "name": f"basic_rooms_{seed}",
        "template": ["".join(r) for r in rows],
        "start": (1, 7),
        "key": (7, 7),
        "terminals": {(6, 1): "signal: sealed-off area beyond the door"},
    }
    return GridWorld(spec, data_dir)


def make_world_general(seed: int = 2, data_dir: str = "data") -> List[GridWorld]:
    """Held-out layouts (never used for training) to measure generalization."""
    side = 9
    base = _render_base(side)
    for y in range(1, side - 1):
        base[y][4] = '#'
    base[2][4] = DOOR
    base[1][6] = EXIT
    base[6][2] = KEY
    base[6][6] = TERMINAL
    specs = {
        "name": f"general_rooms_{seed}",
        "template": ["".join(r) for r in base],
        "start": (1, 1),
        "key": (2, 6),
        "terminals": {(6, 6): "signal: study of unfamiliar space"},
    }

    other = _render_base(side)
    other[1][6] = EXIT
    other[6][6] = KEY
    open_spec = {
        "name": f"general_open_{seed}",
        "template": ["".join(r) for r in other],
        "start": (1, 1),
        "key": (6, 6),
        "terminals": {},
    }
    return [GridWorld(specs, data_dir), GridWorld(open_spec, data_dir)]

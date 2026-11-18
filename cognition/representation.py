"""State representation: encoding of raw observations into agent-usable
discrete signatures / feature dicts.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

DIRS: Dict[str, Tuple[int, int]] = {
    "north": (0, -1), "south": (0, 1), "east": (1, 0), "west": (-1, 0),
}

ACTIONS = ["north", "south", "east", "west", "interact", "read_terminal",
           "write_file", "load_program", "run_program", "wait"]


def encode_state(obs: dict) -> str:
    view_agent = obs["local_view"][1][1]
    view_str = "|".join("".join(r) for r in obs["local_view"])
    return f"inv={1 if 'key' in obs['inventory'] else 0}|here={view_agent}|v={view_str}"


def encode_state_key(obs: dict) -> Tuple[str, ...]:
    """Compact fully-observed signature: absolute position + tile under the
    agent + 4-neighbor occupancy + inventory."""
    view = obs["local_view"]
    agent_view = (view[0][1], view[1][0], view[1][2], view[2][1])  # N,W,E,S
    x, y = obs["agent"]
    return (str(x), str(y), view[1][1], "".join(agent_view),
            obs["inventory"][0] if obs["inventory"] else "")


def wall_signature(local_view: List[List[str]]) -> Tuple[bool, bool, bool, bool]:
    """(blocked_N, blocked_W, blocked_E, blocked_S) from 3x3 window."""
    blocked_N = local_view[0][1] == '#'
    blocked_W = local_view[1][0] == '#'
    blocked_E = local_view[1][2] == '#'
    blocked_S = local_view[2][1] == '#'
    return blocked_N, blocked_W, blocked_E, blocked_S

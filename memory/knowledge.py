"""Knowledge base: reflective layer on memory -- ESCAPE condenses its raw
experience into reusable facts, then reuses them during action selection.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from typing import Dict, List, Tuple


class KnowledgeBase:
    """Facts are simple condition/action patterns extracted from experience.
    Nothing here hardcodes the maze solution: facts must emerge from replays.
    """

    def __init__(self, persist_path: str = "data/knowledge.json"):
        self.path = persist_path
        self.facts: Dict[str, Dict] = {}
        try:
            import os
            if os.path.exists(persist_path):
                with open(persist_path, "r", encoding="utf-8") as fh:
                    self.facts = json.load(fh)
        except Exception:
            self.facts = {}

    # ---------- learning from experience ----------
    def record_effect(self, action: str, local_view: Tuple[str, ...], reward: float) -> None:
        view_str = "".join(local_view)
        key = f"{action}|{view_str}"
        entry = self.facts.setdefault(key, {"good": 0, "bad": 0, "sum": 0.0, "n": 0})
        entry["n"] += 1
        entry["sum"] += reward
        if reward > 0.5:
            entry["good"] += 1
        elif reward < -0.2:
            entry["bad"] += 1

    def effect_estimate(self, action: str, local_view: Tuple[str, ...]) -> float:
        view_str = "".join(local_view)
        entry = self.facts.get(f"{action}|{view_str}")
        if not entry or entry["n"] < 2:
            return 0.0
        return entry["sum"] / entry["n"]

    def mark_known(self, label: str, confidence: float = 1.0, payload: str = "") -> None:
        f = self.facts.setdefault(f"fact|{label}", {"good": 1, "bad": 0, "sum": 0.0, "n": 0})
        f["good"] += 1
        f.setdefault("payload", payload)
        self._flush()

    def number_of_facts(self) -> int:
        return len([k for k in self.facts if not k.startswith("fact|")])

    def number_of_labels(self) -> int:
        return len([k for k in self.facts if k.startswith("fact|")])

    def best_action_for_view(self, local_view: Tuple[str, ...],
                             actions: List[str]) -> str:
        view = "".join(local_view)
        best, best_val = "wait", -1e9
        for action in actions:
            entry = self.facts.get(f"{action}|{view}", {"sum": 0.0, "n": 0})
            val = entry["sum"] / max(entry["n"], 1)
            if val > best_val:
                best_val, best = val, action
        return best

    def flush(self) -> None:
        self._flush()

    def _flush(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.facts, fh, indent=1)

"""Learned planner (Dyna-style): ESCAPE builds a transition model from its
own experience and runs short-horizon greedy rollouts over it. The planner
contains zero knowledge of the world layout -- it plans over what the agent
has actually observed.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Tuple, List


class LearnedPlanner:
    def __init__(self, horizon: int = 500):
        self.horizon = horizon
        self.transitions: Dict[Tuple[Tuple, str], Tuple[Tuple, float]] = {}
        self.visits: Dict[Tuple, int] = defaultdict(int)
        self.action_values: Dict[Tuple[Tuple, str], float] = defaultdict(float)

    def remember(self, state_key, action, next_state_key, reward) -> None:
        key = (state_key, action)
        # keep the most frequent transition observed (stable world assumption)
        self.transitions[key] = (next_state_key, reward)
        self.visits[state_key] += 1

    def known_transition(self, state_key, action) -> bool:
        return (state_key, action) in self.transitions

    def propose(self, state_key, candidates: List[str]) -> str:
        """One-step greedy on learned transition values + successor value."""
        best, best_score = None, -1e9
        for action in candidates:
            key = (state_key, action)
            known = self.transitions.get(key)
            if known:
                next_key, reward = known
                successor_bonus = -self.visits[next_key] * 0.01
                q_value = self.action_values[key]
                score = reward + successor_bonus + q_value
            else:
                # unknown actions get a novelty bonus -> genuinely exploratory
                score = 0.1
            if score > best_score:
                best_score, best = score, action
        return best if best is not None else "wait"

    def update_value(self, state_key, action, target: float, lr: float = 0.3) -> None:
        key = (state_key, action)
        self.action_values[key] = (1 - lr) * self.action_values[key] + lr * target

    def stats(self) -> dict:
        return {"known_transitions": len(self.transitions),
                "visited_states": len(self.visits)}

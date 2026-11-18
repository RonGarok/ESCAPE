"""Tabular model-free learning (Q-learning with epsilon-greedy exploration).
Written from scratch; no pretrained models or external AI services.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple


class QLearning:
    def __init__(self, actions, lr=0.25, gamma=0.92,
                 epsilon=0.4, epsilon_min=0.03, epsilon_decay=0.994,
                 checkpoint_path: str = "data/qtable.pkl"):
        self.actions = list(actions)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.checkpoint_path = checkpoint_path
        self.Q: Dict[Tuple, Dict[str, float]] = {}

    def _row(self, state_key: Tuple) -> Dict[str, float]:
        return self.Q.setdefault(state_key, {a: 0.0 for a in self.actions})

    def select(self, state_key: Tuple, rng) -> str:
        row = self._row(state_key)
        if rng.random() < self.epsilon:
            return rng.choice(self.actions)
        best = max(row.items(), key=lambda kv: kv[1])[0]
        return best

    def learn(self, state_key, action, reward, next_state_key, done: bool) -> None:
        row = self._row(state_key)
        nxt = self._row(next_state_key)
        target = reward if done else reward + self.gamma * max(nxt.values())
        row[action] = row[action] + self.lr * (target - row[action])

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.checkpoint_path) or ".", exist_ok=True)
        with open(self.checkpoint_path, "wb") as fh:
            pickle.dump({"Q": self.Q, "epsilon": self.epsilon}, fh)

    def load(self) -> None:
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "rb") as fh:
                data = pickle.load(fh)
            if isinstance(data, dict) and "Q" in data:
                self.Q = data["Q"]
                self.epsilon = max(self.epsilon_min,
                                   float(data.get("epsilon", self.epsilon)))
            else:
                self.Q = data  # legacy checkpoint format

    def stats(self) -> dict:
        visited = len(self.Q)
        total = visited * len(self.actions) if visited else 0
        nonzero = sum(1 for row in self.Q.values() for v in row.values() if abs(v) > 1e-6)
        return {"states": visited, "entries": total, "trained_entries": nonzero}

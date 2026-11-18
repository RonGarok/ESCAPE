"""Pattern detection: ESCAPE learns event correlations (e.g. "bumping into
D often follows a failed open with empty inventory") purely from statistics
over its own experience. Nothing solution-specific is hardcoded.
"""
from __future__ import annotations

from collections import defaultdict, Counter
from typing import Dict, List, Tuple


class PatternDetector:
    def __init__(self):
        self._after_counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._reward_after: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        self.failed_before: Dict[Tuple[str, str], int] = defaultdict(int)
        self.sequence_history: List[Tuple[str, str]] = []

    def remember_transition(self, state_key: Tuple, action: str,
                            reward: float, next_state_key: Tuple) -> None:
        self._after_counter[(state_key, action)][next_state_key] += 1
        self._reward_after[(state_key, action)].append(reward)
        self.sequence_history.append((state_key, action))
        if len(self.sequence_history) > 200:
            self.sequence_history.pop(0)

    def remember_failure(self, state_key: Tuple, action: str) -> None:
        self.failed_before[(state_key, action)] += 1

    def mean_reward(self, state_key: Tuple, action: str) -> float:
        rewards = self._reward_after.get((state_key, action))
        if not rewards:
            return 0.0
        return sum(rewards) / len(rewards)

    def outcome_probabilities(self, state_key: Tuple, action: str) -> Dict[Tuple, float]:
        counts = self._after_counter.get((state_key, action))
        if not counts:
            return {}
        total = sum(counts.values())
        return {k: c / total for k, c in counts.items()}

    def repeated_failure(self, state_key: Tuple, action: str, min_count: int = 3) -> bool:
        return self.failed_before[(state_key, action)] >= min_count

    def suggest_alternative(self, state_key: Tuple, tried: str,
                            candidates: List[str]) -> str:
        best, best_r = tried, -1e9
        for a in candidates:
            if a == tried:
                continue
            r = self.mean_reward(state_key, a)
            if r > best_r:
                best_r, best = r, a
        return best

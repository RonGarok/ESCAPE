"""Performance measurement: success rate, steps-to-goal, error rate,
retention, planning utility. Used by evaluation AND by evolution to decide
whether a cognitive modification actually improves the agent.
"""
from __future__ import annotations

from typing import Dict, List


def summarize(episodes: List[Dict]) -> Dict:
    if not episodes:
        return {"n": 0, "success_rate": 0.0, "avg_reward": 0.0,
                "avg_steps": 0.0, "error_rate": 0.0}
    n = len(episodes)
    success = sum(1 for e in episodes if e.get("success")) / n
    rewards = sum(e.get("reward", 0.0) for e in episodes) / n
    steps = sum(e.get("steps", 0) for e in episodes) / n
    errors = sum(e.get("bumps", 0) for e in episodes) / max(
        sum(e.get("steps", 1) for e in episodes), 1)
    return {"n": n, "success_rate": success, "avg_reward": rewards,
            "avg_steps": steps, "error_rate": errors}


def retention(before: List[Dict], after: List[Dict]) -> float:
    """How much of a previously learned skill is preserved after more training
    or after a cognitive modification (0..1)."""
    b, a = summarize(before), summarize(after)
    if b["success_rate"] <= 0.0:
        return 0.0
    return min(1.0, a["success_rate"] / b["success_rate"])


def planning_score(with_planner: List[Dict], without: List[Dict]) -> float:
    w, wo = summarize(with_planner), summarize(without)
    denom = wo["avg_steps"] or 1.0
    return max(0.0, (denom - w["avg_steps"]) / denom)


def error_avoidance(episodes_by_stage: List[List[Dict]]) -> List[float]:
    out = []
    for episodes in episodes_by_stage:
        s = summarize(episodes)
        out.append(s["error_rate"])
    return out

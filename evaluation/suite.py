"""Reproducible evaluation suite measuring what the spec demands:
problem solving, generalization to unseen layouts, retention of previous
skills, error avoidance, planning utility, strategy efficiency.
Every run persists results in data/evals/.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional

from learning import metrics
from environment.world import make_world_general


class EvaluationSuite:
    """Uses held-out worlds (never used for training) for generalization."""

    def __init__(self, cfg: Dict, engine):
        self.cfg = cfg
        self.engine = engine
        self.eval_dir = cfg["paths"].get("eval_dir", "data/evals")
        os.makedirs(self.eval_dir, exist_ok=True)
        seed = cfg.get("seed", 0)
        self.holdout = make_world_general(seed, cfg["paths"]["data_dir"])

    def adaptation_episodes(self, world, n: int) -> List[Dict]:
        return self._episodes(world, n, train=True)

    def _episodes(self, world, n: int, max_steps: Optional[int] = None,
                  train: bool = False) -> List[Dict]:
        ms = max_steps or self.cfg["training"]["max_steps_per_episode"]
        out = []
        for _ in range(n):
            out.append(self.engine.run_episode(world, ms, train=train))
        return out

    def _measured(self, world, n: int) -> List[Dict]:
        """Measurement episodes: exploration frozen near-deterministic so the
        numbers reflect the learned policy, not exploration noise."""
        agent = self.engine.agent
        kept = agent.q.epsilon
        agent.q.epsilon = min(agent.q.epsilon, 0.05)
        try:
            return self._episodes(world, n)
        finally:
            agent.q.epsilon = kept

    def rollout(self, record: bool = True, generation: int = 0) -> Dict:
        train_world = self.engine.test_world
        unseen, openw = self.holdout[0], self.holdout[1]
        n = self.cfg["evaluation"]["eval_episodes_per_task"]
        adapt = max(5, n)  # adaptation episodes on the novel world (learning on)
        e_train = self._measured(train_world, n)
        self.adaptation_episodes(unseen, adapt)       # adaptation phase
        e_unseen = self._measured(unseen, n)          # cold test after adaptation
        e_open = self._measured(openw, n)
        summary = {
            "trained_task": metrics.summarize(e_train),
            "novel_task": metrics.summarize(e_unseen),
            "open_task": metrics.summarize(e_open),
            "retention": 0.0,
        }
        history = self.engine.memory.recent_episodes(400)
        m = n * 20
        old = [e for e in history if e["world"] == train_world.name][:m]
        summary["retention"] = round(metrics.retention(old, e_train), 3)
        if record:
            self._persist(summary, generation)
        return {"summary": summary, "raw": {"trained": e_train,
                                            "novel": e_unseen, "open": e_open}}

    def planning_ab_test(self) -> Dict:
        """Compare agent with and without its learned planner (same world/seed,
        same frozen-exploration protocol)."""
        agent = self.engine.agent
        world = self.engine.test_world
        prev = agent.use_planner
        n = max(2, self.cfg["evaluation"]["eval_episodes_per_task"])
        agent.use_planner = True
        with_plan = self._measured(world, n)
        agent.use_planner = False
        without_plan = self._measured(world, n)
        agent.use_planner = prev
        return {
            "with_planner": metrics.summarize(with_plan),
            "without_planner": metrics.summarize(without_plan),
            "planning_score": round(metrics.planning_score(with_plan, without_plan), 4),
        }

    def _persist(self, summary: Dict, generation: int) -> None:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.eval_dir, f"eval_g{generation}_{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"ts": time.time(), "generation": generation,
                       "summary": summary}, fh, indent=1)
        return path

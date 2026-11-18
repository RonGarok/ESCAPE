"""Self-development: candidate cognitive configurations ("genes") are
proposed, trained on the same curriculum, and accepted ONLY if they beat the
current champion on the reproducible evaluation suite. Losers are rolled
back. Each accepted change is versioned in data/evolution/versions.json.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from typing import Dict, List

MUTABLE = {
    "learning_rate": (0.05, 0.6),
    "gamma": (0.5, 0.995),
    "epsilon_start": (0.1, 0.6),
    "epsilon_decay": (0.98, 0.9995),
    "novelty_bonus": (0.0, 0.2),
}


class Genome:
    def __init__(self, cognition: Dict):
        self.cognition = dict(cognition)
        self.id = "c0"
        self.fitness = None

    def mutate(self, rng: random.Random) -> "Genome":
        child = copy.deepcopy(self.cognition)
        key = rng.choice(list(MUTABLE) + ["use_planner", "use_patterns"])
        if key in ("use_planner", "use_patterns"):
            child[key] = not bool(child.get(key, False))
        else:
            lo, hi = MUTABLE[key]
            val = child[key] * rng.uniform(0.7, 1.4)
            child[key] = round(min(hi, max(lo, val)), 5)
        g = Genome(child)
        g.id = f"c{rng.randrange(10**6):06d}"
        return g


class EvolutionOptimizer:
    def __init__(self, cfg: Dict, engine_factory):
        self.cfg = cfg
        self.ev_cfg = cfg["evolution"]
        self.engine_factory = engine_factory
        self.versions_path = os.path.join(cfg["paths"]["evolution_dir"], "versions.json")
        os.makedirs(cfg["paths"]["evolution_dir"], exist_ok=True)
        self.history: List[Dict] = []
        if os.path.exists(self.versions_path):
            with open(self.versions_path, "r", encoding="utf-8") as fh:
                self.history = json.load(fh)

    def _record(self, entry: Dict) -> None:
        self.history.append(entry)
        with open(self.versions_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=1)

    def rollout_fitness(self, cognition: Dict) -> Dict:
        """Train a candidate from scratch on identical seeds and evaluate."""
        trial_cfg = copy.deepcopy(self.cfg)
        trial_cfg["cognition"].update(cognition)
        trial_cfg["training"]["episodes_per_generation"] = max(
            30, self.cfg["training"]["episodes_per_generation"] // 2)
        engine = self.engine_factory(trial_cfg, fresh_agent=True)
        engine.train(generations=1)
        result = engine.evaluator.rollout(record=False, generation=0)
        engine.close()
        s = result["summary"]
        return {
            "success_rate": (s["trained_task"]["success_rate"]
                             + s["novel_task"]["success_rate"]
                             + s["open_task"]["success_rate"]) / 3.0,
            "avg_reward": (s["trained_task"]["avg_reward"]
                           + s["novel_task"]["avg_reward"]
                           + s["open_task"]["avg_reward"]) / 3.0,
        }

    def improve(self) -> Dict:
        champion_cfg = dict(self.cfg["cognition"])
        champion = self.rollout_fitness(champion_cfg)
        report = {"champion_start": champion, "accepted": [], "rejected": []}
        for _ in range(self.ev_cfg["candidates_per_generation"]):
            candidate = Genome(champion_cfg).mutate(random.Random(os.urandom(4)))
            trials = []
            for _ in range(self.ev_cfg["trials_per_candidate"]):
                trials.append(self.rollout_fitness(candidate.cognition))
            avg = {
                "success_rate": sum(t["success_rate"] for t in trials) / len(trials),
                "avg_reward": sum(t["avg_reward"] for t in trials) / len(trials),
            }
            if (avg["success_rate"], avg["avg_reward"]) > (
                    champion["success_rate"], champion["avg_reward"]):
                champion, champion_cfg = avg, candidate.cognition
                report["accepted"].append({"id": candidate.id,
                                           "genes": candidate.cognition, "fitness": avg})
                self._record({"ts": time.time(), "event": "accepted",
                              "id": candidate.id, "genes": candidate.cognition,
                              "fitness": avg})
            else:
                report["rejected"].append({"id": candidate.id,
                                           "genes": candidate.cognition, "fitness": avg})
                self._record({"ts": time.time(), "event": "rejected",
                              "id": candidate.id, "genes": candidate.cognition,
                              "fitness": avg})
        report["champion_final"] = {"genes": champion_cfg, "fitness": champion}
        return report

    def rollback_last(self) -> Dict:
        if not self.history:
            return {"rolled_back": False}
        last = self.history.pop()
        with open(self.versions_path, "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=1)
        return {"rolled_back": True, "dropped": last}

"""Orchestrator: binds agent, environment, memory, evaluator and evolution
into the 10-phase autonomous loop described in the project spec:
observe, analyse, hypothesize, set subgoal, act, execute, observe result,
evaluate, learn, memorize.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Optional

from core.agent import Agent
from environment.world import GridWorld, make_world_basic, make_world_general
from environment.vfs import VirtualFS
from memory.store import MemoryStore
from memory.knowledge import KnowledgeBase
from evaluation.suite import EvaluationSuite


class Engine:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.rng = random.Random(cfg.get("seed", 0))
        paths = cfg["paths"]
        os.makedirs(paths["logs_dir"], exist_ok=True)
        self.memory = MemoryStore(paths["memory_db"])
        self.knowledge = KnowledgeBase(os.path.join(paths["data_dir"], "knowledge.json"))
        self.fs = VirtualFS(os.path.join(paths["data_dir"], "vfs"))
        self.test_world = make_world_basic(cfg.get("seed", 0), paths["data_dir"])
        self.test_world.fs = self.fs
        self.test_world.set_seed(cfg.get("seed", 0))
        self.agent = Agent(cfg, memory=self.memory, knowledge=self.knowledge)
        self.agent.rng = self.rng
        self.evaluator = EvaluationSuite(cfg, self)
        self.generation = 0
        self.started = time.time()
        self.transparent = cfg.get("interface", {}).get("obs_mode", False)
        self.logfile = os.path.join(paths["logs_dir"], "escape.log")
        self._log("engine initialized")

    # ---------- logging ----------
    def _log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        with open(self.logfile, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if self.transparent:
            print(line)

    # ---------- curriculum ----------
    def training_worlds(self, generation: int) -> List[GridWorld]:
        worlds = [self.test_world]
        if generation >= 1:
            worlds += make_world_general(100 + generation, self.cfg["paths"]["data_dir"])
            for w in worlds[1:]:
                w.fs = self.fs
                w.set_seed(100 + generation)
        return worlds

    # ---------- the 10-phase loop ----------
    def run_episode(self, world: GridWorld, max_steps: int, train: bool = True) -> Dict:
        obs = world.reset()
        state_key = self.agent.observe(obs)
        total_r, steps, success = 0.0, 0, False
        for _ in range(max_steps):
            action = self.agent.act(obs)
            if self.transparent:
                self._log(f"state={state_key} -> {action}")
            next_obs, reward, done = world.step(action)
            if train:
                self.agent.learn_step(obs, action, reward, next_obs, done)
            total_r += reward
            steps += 1
            obs = next_obs
            if done:
                success = True
                break
        bumps = self.agent.episode_bumps
        summary = self.agent.end_episode(world.name, total_r, steps, success,
                                         generation=self.generation)
        return {
            "world": world.name, "success": success, "reward": total_r,
            "steps": steps, "bumps": bumps, **summary,
        }

    def train(self, generations: Optional[int] = None,
              progress_cb=None) -> Dict:
        gens = generations or self.cfg["training"]["generations"]
        per_gen = self.cfg["training"]["episodes_per_generation"]
        max_steps = self.cfg["training"]["max_steps_per_episode"]
        results = {"generations": []}
        for g in range(gens):
            self.generation = g
            gen_results = []
            for i in range(per_gen):
                world = self.rng.choice(self.training_worlds(g))
                result = self.run_episode(world, max_steps)
                gen_results.append(result)
                if progress_cb:
                    progress_cb(g, i, result)
            self.agent.save()
            roll = self.evaluator.rollout(record=True, generation=g)
            results["generations"].append({
                "generation": g, "episodes": gen_results, "eval": roll,
            })
            self._log(f"generation {g} done: eval={roll['summary']}")
        return results

    # ---------- status / interface ----------
    def status(self) -> Dict:
        q_stats = self.agent.q.stats()
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "episodes": self.memory.episode_count(),
            "facts": self.knowledge.number_of_facts() + self.knowledge.number_of_labels(),
            "q_states": q_stats["states"],
            "trained_entries": q_stats["trained_entries"],
            "planner_transitions": self.agent.planner.stats()["known_transitions"],
            "success_rate_recent": round(self.memory.success_rate(last_n=30), 3),
        }

    def snapshot_save(self, tag: str) -> None:
        folder = self.cfg["paths"]["evolution_dir"]
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"snapshot_{tag}.pkl"), "wb") as fh:
            import pickle
            pickle.dump(self.agent.snapshot(), fh)

    def snapshot_restore(self, tag: str) -> None:
        src = os.path.join(self.cfg["paths"]["evolution_dir"], f"snapshot_{tag}.pkl")
        if os.path.exists(src):
            with open(src, "rb") as fh:
                import pickle
                self.agent.restore(pickle.load(fh))

    def close(self) -> None:
        self.agent.save()
        self.memory.close()

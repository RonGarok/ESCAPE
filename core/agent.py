"""Core agent: perception -> hypothesis -> subgoal -> action -> learn.

The agent starts nearly blank: it knows the action verbs, not how to solve
anything. Its competence emerges from Q-learning + pattern statistics +
its own learned planner.
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, Tuple

from cognition.representation import ACTIONS, encode_state_key
from cognition.patterns import PatternDetector
from cognition.planner import LearnedPlanner
from learning.q_learning import QLearning
from learning import metrics


class Agent:
    def __init__(self, cfg: Dict, memory=None, knowledge=None):
        cog = cfg["cognition"]
        self.rng = None  # set by engine (deterministic seed)
        self.memory = memory
        self.knowledge = knowledge
        self.actions = list(ACTIONS)
        self.q = QLearning(
            actions=self.actions,
            lr=cog["learning_rate"], gamma=cog["gamma"],
            epsilon=cog["epsilon_start"], epsilon_min=cog["epsilon_min"],
            epsilon_decay=cog["epsilon_decay"],
            checkpoint_path=os.path.join(cfg["paths"]["data_dir"], "qtable.pkl"),
        )
        self.q.load()
        self.patterns = PatternDetector()
        self.planner = LearnedPlanner(horizon=cog["planner_horizon"])
        self.use_planner = cog["use_planner"]
        self.use_patterns = cog["use_patterns"]
        self.novelty_bonus = cog["novelty_bonus"]
        self.last_state_key: Tuple = ()
        self.last_action: str = "wait"
        self.episode_bumps = 0

    # ------- perception / decision -------
    def observe(self, obs: dict) -> Tuple:
        state_key = encode_state_key(obs)
        self.last_state_key = state_key
        return state_key

    def decide(self, state_key: Tuple) -> str:
        row = self.q._row(state_key)
        if self.rng.random() < self.q.epsilon:
            # directed exploration: prefer the learned-transition model when
            # available, otherwise uniform random
            if self.use_planner and self.rng.random() < 0.6:
                return self.planner.propose(state_key, self.actions)
            return self.rng.choice(self.actions)
        action = max(row.items(), key=lambda kv: kv[1])[0]
        if self.use_patterns and self.patterns.repeated_failure(state_key, action, 3):
            alt = self.patterns.suggest_alternative(state_key, action, self.actions)
            if alt != action and self.patterns.mean_reward(state_key, alt) > \
                    self.patterns.mean_reward(state_key, action):
                return alt
        return action

    def act(self, obs: dict) -> str:
        key = self.observe(obs)
        self.last_action = self.decide(key)
        return self.last_action

    # ------- feedback / learning -------
    def learn_step(self, obs, action, reward, next_obs, done: bool) -> None:
        next_key = encode_state_key(next_obs)
        if reward <= -0.2:
            self.episode_bumps += 1
        self.patterns.remember_transition(self.last_state_key, action, reward, next_key)
        if reward <= -0.2:
            self.patterns.remember_failure(self.last_state_key, action)
        shaped = reward + self._novelty(next_key)
        self.q.learn(self.last_state_key, action, shaped, next_key, done)
        self.planner.remember(self.last_state_key, action, next_key, shaped)
        succeeded = shaped > 0.5
        self.planner.update_value(
            self.last_state_key, action,
            shaped + (0 if done else max(self.planner.action_values.get((next_key, a), 0.0)
                                          for a in self.actions)),
        )
        if self.memory is not None:
            self.memory.log_experience(
                world=next_obs.get("world", "?"),
                state_key=str(self.last_state_key),
                action=action, reward=reward,
                next_key=str(next_key), done=done,
            )
        if self.knowledge is not None:
            self.knowledge.record_effect(action, self.last_state_key, reward)
        self.last_state_key = next_key

    def _novelty(self, next_state_key: Tuple) -> float:
        visits = self.planner.visits[next_state_key]
        if visits == 0:
            return self.novelty_bonus
        return self.novelty_bonus / (1 + visits)

    def end_episode(self, world_name: str, reward: float, steps: int, success: bool,
                    generation: int = 0) -> Dict:
        self.q.decay_epsilon()
        summary = metrics.summarize([{
            "success": success, "reward": reward, "steps": steps,
            "bumps": self.episode_bumps,
        }])
        if self.memory is not None:
            self.memory.log_episode(world=world_name, reward=reward, steps=steps,
                                    success=success, generation=generation)
        if self.knowledge is not None:
            self.knowledge.flush()
        if success and self.knowledge is not None:
            self.knowledge.mark_known(f"exit_reached::{world_name}",
                                      payload={"steps": steps})
        self.episode_bumps = 0
        return summary

    # ------- hypothesis / goal hooks (transparent interface) -------
    def hypotheses(self) -> Dict:
        top = sorted(
            ((k, self.planner.action_values[k]) for k in self.planner.action_values),
            key=lambda kv: kv[1], reverse=True,
        )[:5]
        return {
            "goals": [f"repeat learned move {k[1]} near {k[0]}" for k, _ in top],
            "observed_facts": self.knowledge.number_of_facts() if self.knowledge else 0,
        }

    def save(self) -> None:
        self.q.save()

    def snapshot(self) -> bytes:
        return pickle.dumps(self.q.Q)

    def restore(self, blob: bytes) -> None:
        self.q.Q = pickle.loads(blob)

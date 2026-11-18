"""Professional terminal interface: live status dashboard + detailed
observation mode (internal states only; never influences decisions).
Falls back to plain text when stdout is not a TTY.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from typing import Dict

CLEAR = "\x1b[2J\x1b[H"
CYAN, GREEN, YELLOW, RED, DIM, RESET = (
    "\x1b[36m", "\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[2m", "\x1b[0m")
BOLD = "\x1b[1m"


def _bar(value: float, width: int = 24) -> str:
    value = max(0.0, min(1.0, value))
    filled = int(value * width)
    return "[" + "#" * filled + "." * (width - filled) + f"] {value:5.1%}"


def render_status(engine, cfg: Dict) -> str:
    s = engine.status()
    lines = []
    lines.append(f"{BOLD}ESCAPE AI — Autonomous Intelligence Research{RESET}")
    lines.append("=" * 62)
    lines.append(f" uptime: {s['uptime_s']}s   episodes: {s['episodes']}   "
                 f"knowledge: {s['facts']} facts")
    lines.append(f" Q-table: {s['q_states']} states / {s['trained_entries']} trained entries")
    lines.append(f" planner: {s['planner_transitions']} learned transitions")
    lines.append(f" recent success rate: {s['success_rate_recent']:5.1%}")
    lines.append("")
    lines.append(f"{BOLD}Recent episodes{RESET}")
    for ep in engine.memory.recent_episodes(8):
        flag = GREEN + "PASS" + RESET if ep["success"] else RED + "FAIL" + RESET
        lines.append(f"  {ep['world']:<22} r={ep['reward']:8.2f} "
                     f"steps={ep['steps']:<4} {flag}")
    lines.append("")
    lines.append(f"{BOLD}Evaluated objectives{RESET}")
    lines.append(f"  memory usage(soft cap): "
                 f"{_bar(min(1.0, s['episodes'] / 5000))}")
    evals = sorted(
        (f for f in os.listdir(cfg["paths"]["eval_dir"]) if f.endswith(".json")))
    if evals:
        with open(os.path.join(cfg["paths"]["eval_dir"], evals[-1]),
                  "r", encoding="utf-8") as fh:
            last = json.load(fh)
        summary = last["summary"]
        lines.append(f"  trained_task  success: "
                     f"{_bar(summary['trained_task']['success_rate'])}")
        lines.append(f"  novel_task    success: "
                     f"{_bar(summary['novel_task']['success_rate'])}")
        lines.append(f"  retention:    {_bar(summary['retention'])}")
        lines.append(f"  planner transitions: "
                     f"{summary['trained_task'].get('avg_steps', 0):.1f} avg steps")
    lines.append("")
    hyp = engine.agent.hypotheses()
    lines.append(f"{BOLD}Hypotheses in play{RESET}")
    for h in hyp["goals"][:3]:
        lines.append(f"  - {h}")
    lines.append(f"{DIM} last eval file: {evals[-1] if evals else '(none)'}{RESET}")
    return "\n".join(lines)


def watch(engine, cfg: Dict, interval: float = 1.0, once: bool = False) -> None:
    if not sys.stdout.isatty():
        once = True
    while True:
        frame = render_status(engine, cfg)
        if once:
            print(frame)
            break
        sys.stdout.write(CLEAR + frame)
        sys.stdout.flush()
        time.sleep(interval)


def render_obs(engine, episode_budget: int = 3, max_steps: int = 60) -> str:
    """Detailed trace of internal state during a few episodes (read-only:
    runs on a throwaway memory store so the persistent log is untouched)."""
    from memory.store import MemoryStore
    kept_memory, kept_ckpt = engine.memory, engine.agent.q.checkpoint_path
    kept_q = engine.agent.snapshot()
    kept_eps = engine.agent.q.epsilon
    engine.memory = MemoryStore(":memory:")
    engine.agent.q.checkpoint_path = ""
    try:
        return _render_obs_inner(engine, episode_budget, max_steps)
    finally:
        engine.memory.close()
        engine.memory = kept_memory
        engine.agent.q.checkpoint_path = kept_ckpt
        engine.agent.restore(kept_q)
        engine.agent.q.epsilon = kept_eps


def _render_obs_inner(engine, episode_budget: int, max_steps: int) -> str:
    out = []
    engine.transparent = True
    for i in range(episode_budget):
        obs = engine.test_world.reset()
        engine.agent.observe(obs)
        out.append(f"--- obs episode {i+1} ---")
        for t in range(max_steps):
            action = engine.agent.act(obs)
            key = engine.agent.last_state_key
            nxt, reward, done = engine.test_world.step(action)
            engine.agent.learn_step(obs, action, reward, nxt, done)
            out.append(
                f" t={t:03d} state={''.join(key)[:9]} act={action:<13} r={reward:+.2f}")
            obs = nxt
            if done:
                out.append(f" === EXIT FOUND at t={t}")
                break
        engine.agent.end_episode(engine.test_world.name, 0, 0, False)
    engine.transparent = False
    return "\n".join(out)

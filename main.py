#!/usr/bin/env python3
"""ESCAPE AI — command-line entry point.

Commands:
    python main.py train [--episodes N] [--generations N] [--obs]
    python main.py eval
    python main.py evolve
    python main.py status
    python main.py obs
    python main.py program  (demo: ESCAPE writes and runs a sandboxed program)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.engine import Engine
from evolution.optimizer import EvolutionOptimizer
from environment.vm import ESCAPEVM, assemble
from environment.vfs import VirtualFS
from interface.dashboard import watch, render_status, render_obs


def load_config(path: str = "config/config.json") -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def engine_factory(cfg: dict, fresh_agent: bool = False):
    cfg = json.loads(json.dumps(cfg))
    cfg = dict(cfg)
    import tempfile
    if fresh_agent:
        tmp = tempfile.mkdtemp(prefix="escape_")
        cfg["paths"] = json.loads(json.dumps(cfg["paths"]))
        cfg["paths"]["memory_db"] = os.path.join(tmp, "memory.db")
        cfg["paths"]["data_dir"] = tmp
        cfg["cognition"]["epsilon_start"] = 0.5
    return Engine(cfg)


def cmd_train(args) -> None:
    cfg = load_config()
    cfg["interface"]["obs_mode"] = bool(args.obs)
    cfg["training"]["episodes_per_generation"] = args.episodes
    cfg["training"]["generations"] = args.generations
    engine = Engine(cfg)
    engine.train(generations=args.generations,
                 progress_cb=lambda g, i, r: print(
                     f"[gen {g}] ep {i+1}: {'PASS' if r['success'] else 'FAIL'}"
                     f" reward={r['reward']:.2f} steps={r['steps']}"))
    print(render_status(engine, cfg))
    engine.close()


def cmd_eval(args) -> None:
    cfg = load_config()
    engine = Engine(cfg)
    result = engine.evaluator.rollout()
    print(json.dumps(result["summary"], indent=2))
    ab = engine.evaluator.planning_ab_test()
    print(json.dumps(ab, indent=2))
    engine.close()


def cmd_evolve(args) -> None:
    cfg = load_config()
    engine = Engine(cfg)
    optimizer = EvolutionOptimizer(cfg, engine_factory)
    report = optimizer.improve()
    print(json.dumps(report, indent=2))
    engine.close()


def cmd_status(args) -> None:
    cfg = load_config()
    engine = Engine(cfg)
    if args.watch:
        watch(engine, cfg)
    else:
        print(render_status(engine, cfg))
    engine.close()


def cmd_obs(args) -> None:
    cfg = load_config()
    engine = Engine(cfg)
    print(render_obs(engine, episode_budget=args.episodes,
                     max_steps=args.max_steps))
    engine.close()


def cmd_program(args) -> None:
    """Demonstrate ESCAPE's self-programming capability inside the sandbox."""
    cfg = load_config()
    fs = VirtualFS(os.path.join(cfg["paths"]["data_dir"], "vfs"))
    source = "\n".join([
        "push 6",
        "push 7",
        "add",
        "print",
        "halt",
    ])
    fs.write("/programs/fib_demo.asm", source)
    vm = ESCAPEVM()
    vm.run(assemble(fs.read("/programs/fib_demo.asm")))
    print("program output:", vm.out, "| steps:", vm.steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="ESCAPE AI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--generations", type=int, default=2)
    p.add_argument("--obs", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("eval")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("evolve")
    p.set_defaults(func=cmd_evolve)

    p = sub.add_parser("status")
    p.add_argument("--watch", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("obs")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=60)
    p.set_defaults(func=cmd_obs)

    sub.add_parser("program").set_defaults(func=cmd_program)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

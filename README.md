# ESCAPE AI — Autonomous Intelligence Research Project

**Start with almost nothing. Learn. Develop. Evolve.**

ESCAPE is an experimental autonomous agent written from scratch in pure Python
(standard library only). It starts with ~no knowledge of its world and must
progressively discover:

- how its environment works (gridworld with walls, keys, locked doors, exit),
- which actions are useful and which cause repeated errors,
- how to plan from its *own learned transition model*,
- and how to improve itself through validated self-modification.

**No LLM. No Ollama. No pre-trained neural model. No external AI service.**
Everything cognitive (Q-learning, pattern detection, planning, knowledge
building, self-evolution) is implemented in this repository and learnable in
principle from scratch by the agent itself.

---

## 1. What is (and is not) simulated

| Layer | Reality |
|---|---|
| Learning (Q-learning, epsilon-greedy) | Real, tabular, from scratch (`learning/q_learning.py`) |
| Planning (Dyna-style) | Real — plans only over transitions the agent has itself observed (`cognition/planner.py`) |
| Pattern detection | Real event-correlation statistics over its own experience (`cognition/patterns.py`) |
| Memory | Real persistent SQLite + JSON (`memory/`) |
| Self-evolution | Real A/B validation: candidates are trained from scratch on identical seeds and only kept if they beat the champion on the reproducible evaluation suite; losses are rolled back (`evolution/optimizer.py`) |
| Environment | Real simulation: gridworld physics + virtual filesystem + a sandboxed bytecode VM (`environment/`) |

ESCAPE has no consciousness and no human-like intentions. It is a reinforcement
learning system with self-development machinery. Nothing in the code encodes
the solution: rewards come only from environment physics (exit +10, key +2,
door +2, step -0.05, bump -0.30).

## 2. Architecture

```
ESCAPE/
├── core/         engine.py (orchestration, 10-phase loop), agent.py (cognitive assembly)
├── cognition/    representation.py, patterns.py, planner.py
├── learning/     q_learning.py (tabular Q-learning), metrics.py
├── memory/       store.py (SQLite experiences/episodes/facts), knowledge.py (facts, JSON)
├── evolution/    optimizer.py (genome, A/B validation, versions, rollback)
├── environment/  world.py (gridworlds), vfs.py (virtual FS), vm.py (sandboxed bytecode VM)
├── evaluation/   suite.py (reproducible eval vs held-out worlds)
├── interface/    dashboard.py (terminal dashboard + observation mode)
├── logs/         escape.log
├── data/         memory.db, knowledge.json, qtable.pkl, evals/, evolution/, vfs/
├── tests/        automated test suite
├── config/       config.json
├── main.py       CLI entry point
├── run.sh        Linux/macOS launcher (creates .venv automatically)
├── run.bat       Windows launcher (creates .venv automatically)
├── requirements.txt
└── README.md
```

### The 10-phase autonomous loop (`core/engine.py::run_episode`)

1. **Observe** — 3x3 local view + inventory (the agent never sees the full map).
2. **Analyse** — state encoded as discrete signature (`cognition/representation.py`).
3. **Hypothesize** — candidate action from Q-table, learned transition model,
   and repeated-failure avoidance statistics.
4. **Subgoal** — implicit sub-goals emerge as learned action-values (find key,
   open door, reach exit) with zero hand-coded rules.
5. **Choose action** — epsilon-greedy over Q + planner proposal.
6. **Execute** — `environment/world.py::step`.
7. **Observe result** — next observation + reward.
8. **Evaluate** — reward, bumps, success, steps.
9. **Learn** — Q-update, planner transition update, pattern update, novelty bonus.
10. **Memorize** — experience row + knowledge fact in persistent storage.

## 3. Environment and isolation

The environment is a *simulation sandbox*, deliberately disconnected from the
host:

- Only a 3×3 window around the agent is observable. The full map is never given.
- **Virtual filesystem** (`environment/vfs.py`): a JSON-backed tree under
  `data/vfs/`. Path traversal (`..`, absolute paths outside the root) raises
  `ValueError`. File size is capped.
- **Sandboxed VM** (`environment/vm.py`): a tiny bytecode interpreter
  (push/load/store/add/sub/cmp/jmp/jz/print/halt/...). There are no I/O
  instructions, no imports, no `exec/eval`, an instruction counter cap
  (20 000 steps), and a bounded stack/registers. A non-terminating program
  cannot hang the host.

**Honest scope remark.** No file-system os-level virtualization is performed;
`data/` is an application subfolder. For the strongest isolation, run ESCAPE
inside a dedicated VM or container (Linux: `docker run -v $PWD:/escape ...`;
Windows: WSL2 or VirtualBox VM). The agent itself has no network code at all,
and the 'escape' objective is an *internal experimental frontier*: reaching the
exit tile of the simulated world. Nothing in this project attempts to touch
host security boundaries.

## 4. Measuring intelligence (`evaluation/suite.py`)

The suite computes, per run and persisted to `data/evals/*.json`:

| Metric | Meaning |
|---|---|
| `trained_task.success_rate` | ability to solve the trained world |
| `novel_task.success_rate` | generalization to a **held-out layout** never trained on |
| `open_task.success_rate` | transfer without locked doors |
| `retention` | preservation of previously learned skill (memory durability) |
| `error_rate` | wall-bump density — regression avoidance |
| `planning_score` | A/B: steps with planner vs. without (same world/seed) |

A code change is **never** counted as an intelligence improvement. Only the
evaluation suite decides, on fixed seeds, reproducibly.

## 5. Self-development (`evolution/optimizer.py`)

For each candidate:

1. mutate a cognitive gene (learning rate, gamma, epsilon schedule, novelty
   bonus, planner/pattern usage toggles),
2. train from scratch on identical seeds in a disposable temp workspace,
3. evaluate on the reproducible suite (multiple trials),
4. accept **only if better** than the current champion; otherwise reject.

All accept/reject events are versioned in
`data/evolution/versions.json`; `rollback_last()` reverts the last change.
A rejected candidate can never silently corrupt the agent: the production
engine is only updated through the validated champion genes.

## 6. Interface

```bash
python main.py status          # one-shot dashboard
python main.py status --watch  # live dashboard (TTY animation)
python main.py obs             # detailed observation mode (internal trace)
python main.py train --episodes 100
python main.py eval            # evaluation suite + planner A/B
python main.py evolve          # self-development cycle
python main.py program         # demo: sandboxed program written & executed
```

The dashboard shows: uptime, episode count, knowledge facts, Q-table size,
planner transitions, recent success rate, last evaluations (trained/novel/
retention), in-play hypotheses, and important events from `logs/escape.log`.
Observation mode prints the internal state→decision→reward trace without ever
influencing the agent's choices.

## 7. Installation

### Windows

```bat
git clone <this repo> ESCAPE
cd ESCAPE
run.bat
:: or manually:
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python main.py train
```

### Linux / macOS

```bash
git clone <this repo> ESCAPE
cd ESCAPE
./run.sh
# or manually:
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py train
```

Python 3.10+ recommended. The runtime itself needs only the standard library;
`pytest` is used for the tests.

## 8. Running the tests

```bash
# Linux/macOS
./run.sh -m pytest tests -q        # or: .venv/bin/python -m pytest tests -q
# Windows
.venv\Scripts\python -m pytest tests -q
```

## 9. Extending ESCAPE

- Add a new world in `environment/world.py::make_world_*` and reference it in
  `core/engine.py::training_worlds` (training) or
  `evaluation/suite.py` (held-out).
- Add a new action verb in `cognition/representation.py::ACTIONS` + a handler
  in `environment/world.py::step`.
- Add a cognitive gene in `evolution/optimizer.py::MUTABLE` — it is then
  automatically evolvable.

## 10. Research honesty

- ESCAPE does **not** claim consciousness, understanding, or intent.
- All metrics are computed from logged episodes; nothing is averaged by hand.
- Regression failure returns a documented `rolled_back` result.
- Per-episode raw data stays in `data/` (SQLite + JSON) for independent audit.

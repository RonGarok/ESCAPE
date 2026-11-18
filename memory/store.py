"""Persistent memory: episodic experience log + structured knowledge base.

SQLite (stdlib only) keeps experiments durable across runs.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Dict, List, Optional


class MemoryStore:
    def __init__(self, db_path: str = "data/memory.db"):
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._create()

    def _create(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS episodes ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " ts REAL, world TEXT, generation INTEGER DEFAULT 0,"
            " reward REAL, steps INTEGER, success INTEGER, notes TEXT)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS experiences ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " world TEXT, state_key TEXT, action TEXT,"
            " reward REAL, next_key TEXT, done INTEGER, ts REAL)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_exp_state ON experiences(state_key)"
        )
        cur.execute(
            "CREATE TABLE IF NOT EXISTS facts ("
            " fact TEXT PRIMARY KEY, kind TEXT, confidence REAL DEFAULT 1.0,"
            " updated REAL, payload TEXT)"
        )
        self.conn.commit()

    # ---------- episodes ----------
    def log_episode(self, world: str, reward: float, steps: int, success: bool,
                    generation: int = 0, notes: str = "") -> None:
        self.conn.execute(
            "INSERT INTO episodes (ts, world, generation, reward, steps, success, notes)"
            " VALUES (?,?,?,?,?,?,?)",
            (time.time(), world, generation, reward, steps, int(success), notes),
        )
        self.conn.commit()

    def recent_episodes(self, limit: int = 20) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT ts, world, generation, reward, steps, success FROM episodes"
            " ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        keys = ("ts", "world", "generation", "reward", "steps", "success")
        return [dict(zip(keys, r)) for r in rows]

    def success_rate(self, world: Optional[str] = None, last_n: int = 50) -> float:
        if world:
            rows = self.conn.execute(
                "SELECT success FROM episodes WHERE world=? ORDER BY id DESC LIMIT ?",
                (world, last_n),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT success FROM episodes ORDER BY id DESC LIMIT ?", (last_n,)
            ).fetchall()
        if not rows:
            return 0.0
        return sum(r[0] for r in rows) / len(rows)

    # ---------- experiences ----------
    def log_experience(self, world: str, state_key: str, action: str,
                       reward: float, next_key: str, done: bool) -> None:
        self.conn.execute(
            "INSERT INTO experiences (world, state_key, action, reward, next_key, done, ts)"
            " VALUES (?,?,?,?,?,?,?)",
            (world, state_key, action, reward, next_key, int(done), time.time()),
        )

    def flush(self) -> None:
        self.conn.commit()

    def similar_experiences(self, state_key: str, limit: int = 10) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT state_key, action, reward FROM experiences WHERE state_key=?"
            " ORDER BY id DESC LIMIT ?", (state_key, limit),
        ).fetchall()
        keys = ("state_key", "action", "reward")
        return [dict(zip(keys, r)) for r in rows]

    def repeated_errors(self, threshold: float = -0.25) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT state_key, action, COUNT(*) c FROM experiences"
            " WHERE reward < ? GROUP BY state_key, action HAVING c >= 2"
            " ORDER BY c DESC LIMIT 20", (threshold,),
        ).fetchall()
        return [{"state_key": r[0], "action": r[1], "count": r[2]} for r in rows]

    # ---------- facts ----------
    def upsert_fact(self, fact: str, kind: str, confidence: float = 1.0, payload: str = "") -> None:
        self.conn.execute(
            "INSERT INTO facts (fact, kind, confidence, updated, payload)"
            " VALUES (?,?,?,?,?)"
            " ON CONFLICT(fact) DO UPDATE SET confidence=?, updated=?, payload=?",
            (fact, kind, confidence, time.time(), payload, confidence, time.time(), payload),
        )
        self.conn.commit()

    def facts(self, kind: Optional[str] = None) -> List[Dict]:
        if kind:
            rows = self.conn.execute(
                "SELECT fact, kind, confidence, payload FROM facts WHERE kind=?", (kind,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT fact, kind, confidence, payload FROM facts"
            ).fetchall()
        return [
            {"fact": r[0], "kind": r[1], "confidence": r[2],
             "payload": json.loads(r[3]) if r[3] else None}
            for r in rows
        ]

    def fact_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def episode_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def close(self) -> None:
        self.conn.close()

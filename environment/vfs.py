"""Virtual filesystem: an in-memory tree persisted to ``data/vfs_state.json``.

ESCAPE can read, write, ls and "execute" programs here. The tree is
namespace-isolated: paths outside the root are rejected, and no file of the
host machine is ever touched.
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional


class VirtualFS:
    MAX_FILE = 4096  # characters

    def __init__(self, root: str = "data", filename: str = "vfs_state.json"):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.path = os.path.join(self.root, filename)
        self.state: Dict[str, Dict] = {}
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as fh:
                self.state = json.load(fh)
        self.state.setdefault("/", {"type": "dir", "children": {}})
        for sub in ("reports", "memory_notes", "programs"):
            self.mkdir("/" + sub)

    # ---------- helpers ----------
    def _norm(self, raw: str) -> str:
        path = "/" + str(raw).strip().lstrip("/")
        parts = []
        for part in path.split("/"):
            if part in ("", "."):
                continue
            if part == "..":
                raise ValueError("traversal outside sandbox")
            parts.append(part)
        return "/" + "/".join(parts)

    def _node(self, path: Optional[str]) -> Dict:
        node: Dict = self.state["/"]
        parts = [p for p in path.split("/") if p]
        for part in parts:
            if node.get("type") != "dir":
                raise FileNotFoundError(path)
            if part not in node["children"]:
                raise FileNotFoundError(path)
            node = node["children"][part]
        return node

    # ---------- mutating ----------
    def mkdir(self, path: str) -> None:
        path = self._norm(path)
        node = self.state["/"]
        for part in [p for p in path.split("/") if p]:
            nxt = node["children"].setdefault(part, {"type": "dir", "children": {}})
            node = nxt
        self.flush()

    def write(self, path: str, content: str) -> None:
        path = self._norm(path)
        content = str(content)[: self.MAX_FILE]
        parts = [p for p in path.split("/") if p]
        node = self.state["/"]
        for part in parts[:-1]:
            nxt = node["children"].setdefault(part, {"type": "dir", "children": {}})
            node = nxt
        node["children"][parts[-1]] = {"type": "file", "content": content}
        self.flush()

    def read(self, path: str) -> str:
        node = self._node(self._norm(path))
        if node.get("type") != "file":
            raise IsADirectoryError(path)
        return node["content"]

    def ls(self, path: str = "/") -> List[str]:
        node = self._node(self._norm(path))
        if node.get("type") == "file":
            return [self._norm(path)]
        keys = list(node["children"].keys())
        return sorted(keys)

    def delete(self, path: str) -> None:
        path = self._norm(path)
        parts = [p for p in path.split("/") if p]
        node = self._node(self._norm("/".join(parts[:-1]) or "/"))
        if parts[-1] in node["children"]:
            del node["children"][parts[-1]]
            self.flush()

    # ---------- persistence / program gating ----------
    def flush(self) -> None:
        os.makedirs(self.root, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=1)

    def autosave(self, world_id: str, step: int, inventory: List[str]) -> None:
        self.state.setdefault("runs", {})[world_id] = {
            "step": step,
            "inventory": list(inventory),
            "note": "autosave marker",
        }
        self.flush()

    def runs_allowed(self, world_id: str) -> bool:
        return world_id in self.state.get("runs", {})

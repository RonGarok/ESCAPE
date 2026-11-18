"""Sandboxed bytecode VM.
Programs given to the VM are list-of-instructions (each instruction a list of
strings). ESCAPE cannot perform I/O, spawn processes, touch the filesystem, or
escape the sandbox. Non-terminating programs are cut off at MAX_STEPS.
"""
from __future__ import annotations

from typing import List

OPS = {
    "push", "load", "store", "add", "sub", "mul", "div", "mod", "abs", "sign",
    "cmp", "lt", "gt", "jmp", "jz", "jnz", "print", "halt", "swap", "dup"
}
ARITH = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}


class ProgramError(Exception):
    pass


class ESCAPEVM:
    MAX_STEPS = 20000
    STACK_SIZE = 512
    REGISTERS = 8
    MAX_MEMORY = 64

    def __init__(self):
        self.registers: List[int] = [0] * self.REGISTERS
        self.stack: List[int] = []
        self.pc = 0
        self.steps = 0
        self.out: List[str] = []
        self.running = True

    def reset(self) -> None:
        self.registers = [0] * self.REGISTERS
        self.stack = []
        self.pc = 0
        self.steps = 0
        self.out = []
        self.running = True

    @property
    def counter_exceeded(self) -> bool:
        return self.steps >= self.MAX_STEPS

    def run(self, program: List[List[str]]) -> List[str]:
        self.reset()
        if not isinstance(program, list):
            raise ProgramError("program must be a list of instructions")
        n = len(program)
        while self.running and self.pc < n and self.steps < self.MAX_STEPS:
            self.steps += 1
            line = program[self.pc]
            if not isinstance(line, list) or len(line) > 6:
                raise ProgramError("bad instruction")
            op = line[0]
            if not isinstance(op, str) or op not in OPS:
                raise ProgramError("unknown op")
            if op == "push":
                if len(line) != 2 or not str(line[1]).lstrip('-').isdigit():
                    raise ProgramError("push needs an int")
                self.stack.append(int(line[1]))
            elif op == "store":
                r = int(line[1]); v = self.stack.pop()
                self.registers[r % self.REGISTERS] = v
            elif op == "load":
                r = int(line[1])
                self.stack.append(self.registers[r % self.REGISTERS])
            elif op in ARITH:
                b = self.stack.pop(); a = self.stack.pop()
                res = ARITH[op](a, b)
                if op == "div":
                    res = a // b if b != 0 else 0
                if op == "mod":
                    res = a % b if b != 0 else 0
                self.stack.append(res if -10**12 < res < 10**12 else 0)
            elif op == "cmp":
                b = self.stack.pop(); a = self.stack.pop()
                self.stack.append((a > b) - (a < b))
            elif op == "lt":
                b = self.stack.pop(); a = self.stack.pop()
                self.stack.append(1 if a < b else 0)
            elif op == "gt":
                b = self.stack.pop(); a = self.stack.pop()
                self.stack.append(1 if a > b else 0)
            elif op == "abs":
                self.stack.append(abs(self.stack[-1]))
            elif op == "sign":
                self.stack.append((self.stack[-1] > 0) - (self.stack[-1] < 0))
            elif op == "jmp":
                self.pc = int(line[1]); self._guard(n); continue
            elif op in ("jz", "jnz"):
                c = self.stack.pop()
                cond = (c == 0) if op == "jz" else (c != 0)
                if cond:
                    self.pc = int(line[1]); self._guard(n); continue
            elif op == "dup":
                self.stack.append(self.stack[-1])
            elif op == "swap":
                a = self.stack.pop(); b = self.stack.pop()
                self.stack.extend([a, b])
            elif op == "print":
                self.out.append(str(self.stack.pop()) if self.stack else "")
            elif op == "halt":
                self.running = False
                break
            self.pc += 1

    def _guard(self, n: int) -> None:
        if self.pc >= n or self.pc < 0:
            raise ProgramError("bad jump target")


def assemble(text: str) -> List[List[str]]:
    prog: List[List[str]] = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prog.append(line.split())
    return prog

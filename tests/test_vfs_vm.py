import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.vfs import VirtualFS
from environment.vm import ESCAPEVM, assemble, ProgramError


class TestVirtualFS(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.fs = VirtualFS(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_read_ls(self):
        self.fs.write("/reports/note.txt", "hello")
        self.assertEqual(self.fs.read("/reports/note.txt"), "hello")
        self.assertIn("reports", self.fs.ls("/"))

    def test_traversal_rejected(self):
        with self.assertRaises(ValueError):
            self.fs.read("../../etc/passwd")

    def test_truncates_large_content(self):
        big = "x" * 20000
        self.fs.write("/reports/big.txt", big)
        self.assertEqual(len(self.fs.read("/reports/big.txt")), 4096)

    def test_persistence(self):
        self.fs.write("/reports/persist.txt", "v")
        fs2 = VirtualFS(self.tmp)
        self.assertEqual(fs2.read("/reports/persist.txt"), "v")

    def test_autosave_and_program_gate(self):
        fs = self.fs
        self.assertFalse(fs.runs_allowed("w1"))
        fs.autosave("w1", 10, ["key"])
        self.assertTrue(fs.runs_allowed("w1"))


class TestVM(unittest.TestCase):
    def setUp(self):
        self.vm = ESCAPEVM()

    def test_arithmetic(self):
        prog = assemble("push 6\npush 7\nadd\nprint\nhalt")
        self.vm.run(prog)
        self.assertEqual(self.vm.out, ["13"])

    def test_infinite_loop_capped(self):
        prog = assemble("push 1\njmp 0")
        self.vm.run(prog)
        self.assertLessEqual(self.vm.steps, ESCAPEVM.MAX_STEPS)

    def test_unknown_op_rejected(self):
        with self.assertRaises(ProgramError):
            self.vm.run(assemble("sudo rm -rf /"))

    def test_bad_jump_rejected(self):
        with self.assertRaises(ProgramError):
            self.vm.run(assemble("jmp 99"))

    def test_memory_isolation(self):
        prog = assemble("push 5\nstore 3\nload 7\nprint\nload 3\nprint\nhalt")
        self.vm.run(prog)
        self.assertEqual(self.vm.out, ["0", "5"])

    def test_conditional(self):
        prog = assemble("push 0\njz 4\npush 111\nprint\npush 222\nprint\nhalt")
        self.vm.run(prog)
        self.assertEqual(self.vm.out, ["222"])

    def test_no_io_apis(self):
        for token in ("open", "import", "exec", "eval"):
            with self.assertRaises(ProgramError):
                self.vm.run(assemble(token + " 1"))


if __name__ == "__main__":
    unittest.main()

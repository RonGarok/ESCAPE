import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main as entry


class TestEngineAndEvolution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_tmp = tempfile.mkdtemp()
        cls.cfg = entry.load_config(os.path.join(
            os.path.dirname(os.path.abspath(entry.__file__)),
            "config", "config.json"))
        cls.cfg["paths"]["data_dir"] = cls.base_tmp
        cls.cfg["paths"]["memory_db"] = os.path.join(cls.base_tmp, "memory.db")
        cls.cfg["paths"]["logs_dir"] = os.path.join(cls.base_tmp, "logs")
        cls.cfg["paths"]["eval_dir"] = os.path.join(cls.base_tmp, "evals")
        cls.cfg["paths"]["evolution_dir"] = os.path.join(cls.base_tmp, "evo")
        # keep tests fast
        cls.cfg["training"]["episodes_per_generation"] = 30
        cls.cfg["training"]["max_steps_per_episode"] = 250
        cls.cfg["training"]["generations"] = 1
        cls.cfg["evaluation"]["eval_episodes_per_task"] = 2

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base_tmp, ignore_errors=True)

    def test_engine_learns_to_exit(self):
        cfg = json.loads(json.dumps(self.cfg))
        engine = entry.Engine(cfg)
        engine.train(generations=2)
        results = engine.evaluator.rollout(record=False)
        s = results["summary"]["trained_task"]
        self.assertGreater(s["success_rate"], 0.0)
        engine.close()

    def test_persistence_across_engines(self):
        cfg = json.loads(json.dumps(self.cfg))
        e1 = entry.Engine(cfg)
        e1.train(generations=1)
        n1 = e1.memory.episode_count()
        e1.close()
        e2 = entry.Engine(cfg)
        self.assertEqual(e2.memory.episode_count(), n1)
        e2.close()

    def test_planning_ab_test_runs(self):
        cfg = json.loads(json.dumps(self.cfg))
        engine = entry.Engine(cfg)
        engine.train(generations=1)
        ab = engine.evaluator.planning_ab_test()
        self.assertIn("planning_score", ab)
        engine.close()

    def test_evolution_rollback(self):
        from evolution.optimizer import EvolutionOptimizer
        cfg = json.loads(json.dumps(self.cfg))
        cfg["evolution"] = {"trials_per_candidate": 1,
                            "candidates_per_generation": 1}
        optimizer = EvolutionOptimizer(cfg, entry.engine_factory)
        report = optimizer.improve()
        self.assertIn("champion_final", report)
        rb = optimizer.rollback_last()
        self.assertTrue(rb["rolled_back"])

    def test_isolation(self):
        from environment.vfs import VirtualFS
        tmp = tempfile.mkdtemp()
        fs = VirtualFS(tmp)
        with self.assertRaises(ValueError):
            fs.read("/../../etc/passwd")
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()

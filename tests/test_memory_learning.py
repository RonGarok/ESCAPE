import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from memory.store import MemoryStore
from memory.knowledge import KnowledgeBase
from learning.q_learning import QLearning
from learning import metrics
from cognition.patterns import PatternDetector
from cognition.planner import LearnedPlanner


class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = MemoryStore(os.path.join(self.tmp, "m.db"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_episode_roundtrip(self):
        self.store.log_episode("w", 1.5, 42, True, generation=0)
        self.store.log_episode("w", -0.3, 99, False, generation=0)
        self.assertEqual(self.store.episode_count(), 2)
        self.assertAlmostEqual(self.store.success_rate("w"), 0.5)
        self.assertEqual(len(self.store.recent_episodes(2)), 2)

    def test_experience_and_errors(self):
        for _ in range(3):
            self.store.log_experience("w", "('s1',)", "south", -0.3, "('s2',)", False)
        errs = self.store.repeated_errors()
        self.assertEqual(len(errs), 1)
        self.assertEqual(errs[0]["count"], 3)

    def test_facts(self):
        self.store.upsert_fact("door_needs_key", "affordance", 0.9, '{"a":1}')
        self.store.upsert_fact("door_needs_key", "affordance", 1.0, '{"a":2}')
        facts = self.store.facts()
        self.assertEqual(len(facts), 1)
        self.assertAlmostEqual(facts[0]["confidence"], 1.0)
        self.assertEqual(facts[0]["payload"]["a"], 2)


class TestKnowledgeBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.kb = KnowledgeBase(os.path.join(self.tmp, "k.json"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_record_and_estimate(self):
        view = ("." * 5)
        self.kb.record_effect("north", view, 1.0)
        self.kb.record_effect("north", view, 1.0)
        self.assertAlmostEqual(self.kb.effect_estimate("north", view), 1.0)
        self.kb.record_effect("west", view, -0.3)
        self.assertEqual(self.kb.best_action_for_view(view, ["north", "west"]), "north")

    def test_persistence(self):
        view = ("." * 5)
        self.kb.record_effect("north", view, 1.0)
        self.kb.record_effect("north", view, 1.0)
        self.kb.flush()
        kb2 = KnowledgeBase(self.kb.path)
        self.assertAlmostEqual(kb2.effect_estimate("north", view), 1.0)


class TestQLearning(unittest.TestCase):
    def test_converges_to_good_action(self):
        import random
        q = QLearning(["a", "b"], lr=0.5, gamma=0.9, epsilon=0.2)
        rng = random.Random(0)
        for _ in range(400):
            s = "s0"
            a = q.select(s, rng)
            r = 1.0 if a == "a" else 0.0
            q.learn(s, a, r, s, False)
        self.assertGreater(q.Q["s0"]["a"], q.Q["s0"]["b"])

    def test_epsilon_decay_floor(self):
        q = QLearning(["a"], epsilon=0.5, epsilon_min=0.05, epsilon_decay=0.5)
        for _ in range(10):
            q.decay_epsilon()
        self.assertAlmostEqual(q.epsilon, 0.05)


class TestPlannerAndMetrics(unittest.TestCase):
    def test_planner_uses_learned_transitions(self):
        p = LearnedPlanner()
        p.remember("s1", "east", "s2", 1.0)
        p.remember("s1", "west", "s3", -0.3)
        self.assertEqual(p.propose("s1", ["east", "west"]), "east")

    def test_metrics_summaries(self):
        eps = [{"success": True, "reward": 5, "steps": 10, "bumps": 1},
               {"success": False, "reward": -1, "steps": 20, "bumps": 3}]
        s = metrics.summarize(eps)
        self.assertAlmostEqual(s["success_rate"], 0.5)
        self.assertAlmostEqual(s["avg_steps"], 15.0)
        self.assertAlmostEqual(metrics.retention(eps, eps), 1.0)


if __name__ == "__main__":
    unittest.main()

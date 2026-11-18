import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.world import (make_world_basic, EXIT_REWARD, KEY_REWARD,
                               BUMP_REWARD, MOVE_REWARD)


class TestWorld(unittest.TestCase):
    def setUp(self):
        self.world = make_world_basic(seed=0, data_dir="data/test_world")
        self.world.set_seed(0)

    def test_square_grid_and_wall_border(self):
        for row in self.world.base_grid:
            self.assertEqual(len(row), 9)
        self.assertEqual(self.world.at(0, 0), '#')
        self.assertEqual(self.world.at(8, 8), '#')

    def test_basic_move(self):
        self.world.reset()
        _, r, _ = self.world.step("north")
        self.assertAlmostEqual(r, MOVE_REWARD)
        self.assertEqual(tuple(self.world.agent), (1, 6))

    def test_wall_collision(self):
        self.world.reset()
        _, r, _ = self.world.step("west")
        self.assertAlmostEqual(r, BUMP_REWARD)
        self.assertEqual(tuple(self.world.agent), (1, 7))

    def test_key_pickup_and_door_open(self):
        self.world.reset()
        self.world.agent = [7, 8]
        self.world.grid[8][7] = 'K'
        _, r, _ = self.world.step("north")
        self.assertAlmostEqual(r, KEY_REWARD)
        self.assertIn('key', self.world.inventory)
        self.world.agent = [6, 2]
        self.world.grid[3][6] = 'D'
        _, r, _ = self.world.step("south")
        self.assertAlmostEqual(r, 2.0)
        self.assertEqual(self.world.grid[3][6], '.')

    def test_door_blocked_without_key(self):
        self.world.reset()
        self.world.agent = [6, 2]
        self.world.grid[3][6] = 'D'
        _, r, _ = self.world.step("south")
        self.assertAlmostEqual(r, -0.30)
        self.assertEqual(self.world.grid[3][6], 'D')

    def test_reach_exit_ends_episode(self):
        self.world.reset()
        self.world.agent = [7, 2]
        _, r, _ = self.world.step("north")
        _, r2, done = self.world.step("interact")
        self.assertTrue(done)
        self.assertAlmostEqual(r2, EXIT_REWARD)

    def test_reset_restores_world(self):
        self.world.reset()
        self.world.agent = [7, 8]
        self.world.grid[8][7] = 'K'
        self.world.step("north")
        self.world.reset()
        self.assertEqual(tuple(self.world.agent), (1, 7))
        self.assertEqual(self.world.inventory, [])
        self.assertEqual(self.world.grid[7][7], 'K')


if __name__ == "__main__":
    unittest.main()

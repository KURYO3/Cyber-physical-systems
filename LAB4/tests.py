import unittest
import numpy as np
from main import simulate

class Tests(unittest.TestCase):
    def setUp(self):
        self.t = np.linspace(0, 10, 101)

    def test_trajectory_shape(self):
        sol = simulate([1.0, 2.0, 3.0], self.t)
        self.assertEqual(sol.shape, (len(self.t), 3))

    def test_same_initial_conditions(self):
        init = [0.5, 0.5, 0.5]
        sol1 = simulate(init, self.t)
        sol2 = simulate(init, self.t)
        diff = np.max(np.abs(sol1 - sol2))
        self.assertTrue(diff < 1e-12)

    def test_no_errors(self):
        try:
            _ = simulate([1,1,1], [0, 1])
        except Exception as e:
            self.fail(f"Помилка симуляції: {e}")

if __name__ == "__main__":
    unittest.main()

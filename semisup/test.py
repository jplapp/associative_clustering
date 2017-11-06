import unittest

import numpy as np

from backend import calc_correct_logit_score


# check calc_correct_logit_score
class TestEvalulationMethods(unittest.TestCase):
    def test_simple(self):
        labels = np.array([1, 1, 0])
        preds = np.array([1, 1, 0])

        conf_mtx, acc = calc_correct_logit_score(preds, labels, 2)

        target = np.array([[1, 0], [0, 2]])

        self.assertEqual(conf_mtx.tolist(), target.tolist())
        self.assertAlmostEqual(acc, 1.0)

    def test_inversed(self):
        labels = np.array([0, 0, 1])
        preds = np.array([1, 1, 0])

        conf_mtx, acc = calc_correct_logit_score(preds, labels, 2)

        target = np.array([[2, 0], [0, 1]])

        self.assertEqual(conf_mtx.tolist(), target.tolist())
        self.assertAlmostEqual(acc, 1.0)

    def test_complex(self):
        labels = np.array([2, 2, 0, 0, 1])
        preds = np.array([1, 1, 0, 2, 2])

        conf_mtx, acc = calc_correct_logit_score(preds, labels, 3)

        target = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 2]])

        self.assertEqual(conf_mtx.tolist(), target.tolist())
        self.assertAlmostEqual(acc, 0.8)


if __name__ == '__main__':
    unittest.main()

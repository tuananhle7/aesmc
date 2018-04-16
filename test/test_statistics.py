import dgm.statistics as stats
import numpy as np
import torch
import torch.nn as nn
import unittest


class TestEmpiricalExpectation(unittest.TestCase):
    def test_dimensions(self):
        value = torch.rand(2, 3, 4, 5, 6)
        log_weight = -torch.rand(2, 3)

        def f(arg):
            return torch.rand(2, 7, 8)

        self.assertEqual(
            stats.empirical_expectation(value, log_weight, f).size(),
            torch.Size([2, 7, 8])
        )

        value = torch.rand(2, 3)
        log_weight = -torch.rand(2, 3)

        def f(arg):
            return torch.rand(2)

        self.assertEqual(
            stats.empirical_expectation(value, log_weight, f).size(),
            torch.Size([2])
        )

    def test_value(self):
        value = torch.Tensor([1, 2, 3]).unsqueeze(0)
        log_weight = torch.log(torch.Tensor([0.2, 0.3, 0.5]).unsqueeze(0))

        def f(value_):
            return value_ * 2

        self.assertTrue(torch.equal(
            stats.empirical_expectation(value, log_weight, f),
            torch.Tensor([1 * 2 * 0.2 + 2 * 2 * 0.3 + 3 * 2 * 0.5])
        ))


class TestLogEss(unittest.TestCase):
    def test_dimensions(self):
        log_weight = -torch.rand(2, 3)
        self.assertEqual(
            stats.log_ess(log_weight).size(),
            torch.Size([2])
        )

        log_weight = -torch.rand(1, 3)
        self.assertEqual(
            stats.log_ess(log_weight).size(),
            torch.Size([1])
        )

        log_weight = -torch.rand(3, 1)
        self.assertEqual(
            stats.log_ess(log_weight).size(),
            torch.Size([3])
        )

        log_weight = -torch.rand(3)
        self.assertEqual(
            stats.log_ess(log_weight).size(),
            torch.Size([])
        )

    def test_value(self):
        normalized_weight = np.array([0.2, 0.3, 0.5])
        log_weight = np.log(normalized_weight * 0.47)
        self.assertAlmostEqual(
            stats.log_ess(torch.from_numpy(log_weight)).item(),
            np.log(1 / np.sum(normalized_weight**2))
        )

        normalized_weight = np.array([0.2, 0.3, 0.5])
        log_weight = np.log(normalized_weight) + 1e6
        self.assertAlmostEqual(
            stats.log_ess(torch.from_numpy(log_weight)).item(),
            np.log(1 / np.sum(normalized_weight**2))
        )

        normalized_weight = np.array([0.2, 0.3, 0.5])
        log_weight = np.log(normalized_weight) - 1e6
        self.assertAlmostEqual(
            stats.log_ess(torch.from_numpy(log_weight)).item(),
            np.log(1 / np.sum(normalized_weight**2))
        )


class TestEss(unittest.TestCase):
    def test_value(self):
        normalized_weight = np.array([0.2, 0.3, 0.5])
        log_weight = np.log(normalized_weight * 0.47)
        self.assertAlmostEqual(
            stats.ess(torch.from_numpy(log_weight)).item(),
            1 / np.sum(normalized_weight**2)
        )

        normalized_weight = np.array([0.2, 0.3, 0.5])
        log_weight = np.log(normalized_weight) + 1e6
        self.assertAlmostEqual(
            stats.ess(torch.from_numpy(log_weight)).item(),
            1 / np.sum(normalized_weight**2)
        )

        normalized_weight = np.array([0.2, 0.3, 0.5])
        log_weight = np.log(normalized_weight) - 1e6
        self.assertAlmostEqual(
            stats.ess(torch.from_numpy(log_weight)).item(),
            1 / np.sum(normalized_weight**2)
        )


if __name__ == '__main__':
    unittest.main()

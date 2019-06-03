import dgm.math as math
import numpy as np
import torch
import unittest


class TestLognormexp(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(
            math.lognormexp(
                torch.rand(2, 3, 4, 5),
                dim=2
            ).size(),
            torch.Size([2, 3, 4, 5])
        )
        self.assertEqual(
            math.lognormexp(torch.rand(3)).size(),
            torch.Size([3])
        )
        self.assertEqual(
            math.lognormexp(torch.rand(1)).size(),
            torch.Size([1])
        )

        self.assertEqual(
            list(np.shape(math.lognormexp(
                np.random.rand(2, 3, 4, 5),
                dim=2
            ))),
            [2, 3, 4, 5]
        )
        self.assertEqual(
            list(np.shape(math.lognormexp(np.random.rand(3)))),
            [3]
        )
        self.assertEqual(
            list(np.shape(math.lognormexp(np.random.rand(1)))),
            [1]
        )

    def test_type(self):
        self.assertIsInstance(
            math.lognormexp(torch.rand(1)),
            torch.Tensor
        )
        self.assertIsInstance(
            math.lognormexp(np.array([2])),
            np.ndarray
        )

    def test_value(self):
        test_input = [1, 2, 3]
        temp = np.exp(1) + np.exp(2) + np.exp(3)
        test_result = np.log(np.exp(test_input) / temp)
        np.testing.assert_allclose(
            math.lognormexp(torch.Tensor(test_input)).numpy(),
            torch.Tensor(test_result).numpy(),
            atol=1e-6
        )
        np.testing.assert_allclose(
            math.lognormexp(np.array(test_input)),
            np.array(test_result),
            atol=1e-6
        )


class TestExponentiateAndNormalize(unittest.TestCase):
    def test_dimensions(self):
        self.assertEqual(
            math.exponentiate_and_normalize(
                torch.rand(2, 3, 4, 5),
                dim=2
            ).size(),
            torch.Size([2, 3, 4, 5])
        )
        self.assertEqual(
            math.exponentiate_and_normalize(torch.rand(3)).size(),
            torch.Size([3])
        )
        self.assertEqual(
            math.exponentiate_and_normalize(torch.rand(1)).size(),
            torch.Size([1])
        )

        self.assertEqual(
            list(np.shape(math.exponentiate_and_normalize(
                np.random.rand(2, 3, 4, 5),
                dim=2
            ))),
            [2, 3, 4, 5]
        )
        self.assertEqual(
            list(np.shape(math.exponentiate_and_normalize(np.random.rand(3)))),
            [3]
        )
        self.assertEqual(
            list(np.shape(math.exponentiate_and_normalize(np.random.rand(1)))),
            [1]
        )

    def test_type(self):
        self.assertIsInstance(
            math.exponentiate_and_normalize(torch.rand(1)),
            torch.Tensor
        )
        self.assertIsInstance(
            math.exponentiate_and_normalize(np.array([2])),
            np.ndarray
        )

    def test_value(self):
        test_input = [1, 2, 3]
        temp = np.exp(1) + np.exp(2) + np.exp(3)
        test_result = [
            np.exp(1) / temp,
            np.exp(2) / temp,
            np.exp(3) / temp,
        ]
        np.testing.assert_allclose(
            math.exponentiate_and_normalize(torch.Tensor(test_input)).numpy(),
            torch.Tensor(test_result).numpy()
        )
        np.testing.assert_allclose(
            math.exponentiate_and_normalize(np.array(test_input)),
            np.array(test_result)
        )


if __name__ == '__main__':
    unittest.main()

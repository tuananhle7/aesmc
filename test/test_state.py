import dgm.state as state
import numpy as np
import torch
import unittest


class TestExpanded(unittest.TestCase):
    def test_expanded(self):
        batch_size, num_particles = 3, 4
        n1 = torch.distributions.Normal(
            loc=torch.zeros(1), scale=torch.ones(1)
        )
        self.assertFalse(state.is_expanded(n1, batch_size, num_particles))

        n2 = torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles),
            scale=torch.ones(batch_size, num_particles)
        )
        self.assertTrue(state.is_expanded(n2, batch_size, num_particles))

        n3 = state.set_expanded(n2, True)
        self.assertEqual(n2, n3)
        self.assertTrue(state.is_expanded(n2))

        state.set_expanded(n2, False)
        self.assertEqual(state.is_expanded(n2), False)


class TestSample(unittest.TestCase):
    def test_dimensions(self):
        # distribution is implicitly NOT expanded
        batch_size, num_particles = 4, 5
        distribution = torch.distributions.Normal(
            loc=torch.zeros(2), scale=torch.ones(2)
        )
        smp = state.sample(distribution, batch_size, num_particles)
        self.assertEqual(
            smp.size(),
            torch.Size([batch_size, num_particles, 2])
        )

        # distribution is implicitly expanded
        batch_size, num_particles = 4, 5
        distribution = torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles),
            scale=torch.ones(batch_size, num_particles)
        )
        smp = state.sample(distribution, batch_size, num_particles)
        self.assertEqual(
            smp.size(),
            torch.Size([batch_size, num_particles])
        )

        # distribution is explicitly NOT expanded
        batch_size, num_particles = 4, 5
        distribution = state.set_expanded(torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles),
            scale=torch.ones(batch_size, num_particles)
        ), False)
        smp = state.sample(distribution, batch_size, num_particles)
        self.assertEqual(
            smp.size(),
            torch.Size([batch_size, num_particles, batch_size, num_particles])
        )

        # distribution is explicitly expanded
        batch_size, num_particles = 4, 5
        distribution = state.set_expanded(torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles),
            scale=torch.ones(batch_size, num_particles)
        ), True)
        smp = state.sample(distribution, batch_size, num_particles)
        self.assertEqual(
            smp.size(),
            torch.Size([batch_size, num_particles])
        )

        # distribution is a dict
        batch_size, num_particles = 4, 5
        distribution = {
            'a': torch.distributions.Normal(
                loc=torch.zeros(2), scale=torch.ones(2)
            ),
            'b': torch.distributions.Normal(
                loc=torch.zeros(3), scale=torch.ones(3)
            )
        }
        smp = state.sample(distribution, batch_size, num_particles)
        self.assertEqual(
            smp['a'].size(),
            torch.Size([batch_size, num_particles, 2])
        )
        self.assertEqual(
            smp['b'].size(),
            torch.Size([batch_size, num_particles, 3])
        )


class TestLogProb(unittest.TestCase):
    def test_dimensions(self):
        batch_size, num_particles = 1, 2
        distribution = torch.distributions.Normal(
            loc=torch.zeros(3, 4), scale=torch.ones(3, 4)
        )
        value = torch.rand(batch_size, num_particles, 3, 4)
        lp = state.log_prob(distribution, value)
        self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))

        batch_size, num_particles = 1, 2
        distribution = torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles, 3, 4),
            scale=torch.ones(batch_size, num_particles, 3, 4)
        )
        value = torch.rand(batch_size, num_particles, 3, 4)
        lp = state.log_prob(distribution, value)
        self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))

        batch_size, num_particles = 1, 2
        distribution = {
            'a': torch.distributions.Normal(
                loc=torch.zeros(3, 4), scale=torch.ones(3, 4)
            ),
            'b': torch.distributions.Normal(
                loc=torch.zeros(5, 6, 7), scale=torch.ones(5, 6, 7)
            )
        }
        value = {
            'a': torch.rand(batch_size, num_particles, 3, 4),
            'b': torch.rand(batch_size, num_particles, 5, 6, 7)
        }
        lp = state.log_prob(distribution, value)
        self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))


class TestLogProbNonReparam(unittest.TestCase):
    def test_dimensions(self):
        batch_size, num_particles = 1, 2
        distribution = torch.distributions.Normal(
            loc=torch.zeros(3, 4), scale=torch.ones(3, 4)
        )
        value = torch.rand(batch_size, num_particles, 3, 4)
        lp = state.log_prob_non_reparam(distribution, value)
        self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))

        batch_size, num_particles = 1, 2
        distribution = torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles, 3, 4),
            scale=torch.ones(batch_size, num_particles, 3, 4)
        )
        value = torch.rand(batch_size, num_particles, 3, 4)
        lp = state.log_prob(distribution, value)
        self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))

        categorical = torch.distributions.Categorical(
            probs=torch.Tensor([0.2, 0.3, 0.5])
        )
        categorical_value = torch.Tensor([
            [1, 0, 2],
            [0, 1, 2]
        ])
        batch_size, num_particles = categorical_value.size()
        normal = torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles, 3, 4),
            scale=torch.ones(batch_size, num_particles, 3, 4)
        )
        normal_value = torch.rand(batch_size, num_particles, 3, 4)
        distribution = {'categorical': categorical, 'normal': normal}
        value = {'categorical': categorical_value, 'normal': normal_value}
        lp = state.log_prob_non_reparam(distribution, value)
        self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))

    def test_value(self):
        batch_size, num_particles = 1, 2
        distribution = torch.distributions.Normal(
            loc=torch.zeros(3, 4), scale=torch.ones(3, 4)
        )
        value = torch.ones(batch_size, num_particles, 3, 4)
        lp = state.log_prob_non_reparam(distribution, value)
        np.testing.assert_equal(
            lp.numpy(), np.zeros([batch_size, num_particles])
        )

        categorical = torch.distributions.Categorical(
            probs=torch.Tensor([0.2, 0.3, 0.5])
        )
        categorical_value = torch.Tensor([
            [1, 0, 2],
            [0, 1, 2]
        ])
        batch_size, num_particles = categorical_value.size()
        normal = torch.distributions.Normal(
            loc=torch.zeros(batch_size, num_particles, 3, 4),
            scale=torch.ones(batch_size, num_particles, 3, 4)
        )
        normal_value = torch.rand(batch_size, num_particles, 3, 4)
        distribution = {'categorical': categorical, 'normal': normal}
        value = {'categorical': categorical_value, 'normal': normal_value}
        lp = state.log_prob_non_reparam(distribution, value)
        np.testing.assert_equal(
            lp.numpy(), state.log_prob(categorical, categorical_value).numpy()
        )


class TestResample(unittest.TestCase):
    def test_dimensions(self):
        ancestral_index = torch.zeros(3, 2).long()
        value = torch.rand(3, 2)
        self.assertEqual(
            value.size(),
            state.resample(value, ancestral_index).size()
        )

        value = torch.rand(3, 2, 4, 5)
        self.assertEqual(
            value.size(),
            state.resample(value, ancestral_index).size()
        )

    def test_small(self):
        ancestral_index = torch.LongTensor(
            [
                [1, 2, 0],
                [0, 0, 1]
            ]
        )
        value = torch.Tensor(
            [
                [1, 2, 3],
                [4, 5, 6]
            ]
        )
        resampled_value = torch.Tensor(
            [
                [2, 3, 1],
                [4, 4, 5]
            ]
        )

        self.assertTrue(torch.equal(
            state.resample(value, ancestral_index),
            resampled_value
        ))


if __name__ == '__main__':
    pass
    #  unittest.main()

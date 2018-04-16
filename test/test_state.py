import dgm.state as state
import numpy as np
import torch
import unittest


class TestBatchShapeMode(unittest.TestCase):
    def test_dimensions(self):
        # Implicit
        batch_size, num_particles = 2, 3
        dim1 = 4
        for batch_shape, inferred_batch_shape_mode, ambiguous in [
            [(), state.DistributionBatchShapeMode.NOT_EXPANDED, False],
            [
                (batch_size,),
                state.DistributionBatchShapeMode.BATCH_EXPANDED,
                True
            ],
            [
                (dim1,),
                state.DistributionBatchShapeMode.NOT_EXPANDED,
                False
            ],
            [
                (batch_size, num_particles),
                state.DistributionBatchShapeMode.FULLY_EXPANDED,
                True
            ],
            [
                (batch_size, dim1),
                state.DistributionBatchShapeMode.BATCH_EXPANDED,
                True
            ],
            [
                (batch_size, num_particles, dim1),
                state.DistributionBatchShapeMode.FULLY_EXPANDED,
                True
            ]
        ]:
            distribution = torch.distributions.Normal(
                loc=torch.zeros(size=batch_shape),
                scale=torch.ones(size=batch_shape)
            )
            if ambiguous:
                with self.assertWarns(RuntimeWarning):
                    self.assertEqual(
                        state.get_batch_shape_mode(
                            distribution, batch_size, num_particles
                        ), inferred_batch_shape_mode
                    )
            else:
                self.assertEqual(
                    state.get_batch_shape_mode(
                        distribution, batch_size, num_particles
                    ), inferred_batch_shape_mode
                )

        # Explicit
        batch_size, num_particles = 2, 3
        dim1 = 4
        batch_shape = (batch_size, num_particles)

        for batch_shape_mode in [
            state.DistributionBatchShapeMode.NOT_EXPANDED,
            state.DistributionBatchShapeMode.BATCH_EXPANDED,
            state.DistributionBatchShapeMode.FULLY_EXPANDED
        ]:
            distribution = state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=torch.zeros(size=batch_shape),
                    scale=torch.ones(size=batch_shape)
                ), batch_shape_mode
            )
            self.assertEqual(
                state.get_batch_shape_mode(
                    distribution, batch_size, num_particles
                ),
                batch_shape_mode
            )


# the following is a temporary implementation for taking sums and means across
# many dimensions until the following is merged:
# https://github.com/pytorch/pytorch/pull/6152
def sum_dims(x, dims, keepdim=False):
    if len(dims) == 0:
        return x
    else:
        original_shape = x.shape
        original_ndimension = x.ndimension()
        for dim in dims:
            x = torch.sum(x, dim=dim, keepdim=True)
        if keepdim:
            return x
        else:
            new_shape = np.array(original_shape)[np.array(
                list(set(range(original_ndimension)) - set(dims))
            )]
            return x.view(*[int(s) for s in new_shape])
        return x


def mean_dims(x, dims, keepdim=False):
    if len(dims) == 0:
        return x
    else:
        original_shape = x.shape
        summed_out_shape = np.array(original_shape)[
            np.array(list(dims))
        ]
        return sum_dims(x, dims, keepdim) / float(np.prod(summed_out_shape))


class TestSample(unittest.TestCase):
    def test_dimensions(self):
        # Implicit
        for batch_size, num_particles in [(2, 2), (2, 3)]:
            for dims in [(), (4,), (4, 5)]:
                for batch_shape, sample_size, ambiguous in [
                    [
                        dims,
                        torch.Size((batch_size, num_particles) + dims),
                        False
                    ],
                    [
                        (batch_size,),
                        torch.Size([batch_size, num_particles]),
                        True
                    ],
                    [
                        (batch_size, num_particles),
                        torch.Size([batch_size, num_particles]),
                        True
                    ],
                    [
                        (batch_size,) + dims,
                        torch.Size((batch_size, num_particles) + dims),
                        True
                    ],
                    [
                        (batch_size, num_particles) + dims,
                        torch.Size((batch_size, num_particles) + dims),
                        True
                    ]
                ]:
                    distribution = torch.distributions.Normal(
                        loc=torch.zeros(size=batch_shape),
                        scale=torch.ones(size=batch_shape)
                    )
                    if ambiguous:
                        with self.assertWarns(RuntimeWarning):
                            self.assertEqual(
                                state.sample(
                                    distribution, batch_size, num_particles
                                ).size(), sample_size
                            )
                    else:
                        self.assertEqual(
                            state.sample(
                                distribution, batch_size, num_particles
                            ).size(), sample_size
                        )

        # Explicit
        batch_size, num_particles = 2, 3
        batch_shape = (batch_size, num_particles)
        distribution = torch.distributions.Normal(
            loc=torch.zeros(size=batch_shape),
            scale=torch.ones(size=batch_shape)
        )
        for batch_shape_mode, sample_size in [
            [
                state.DistributionBatchShapeMode.NOT_EXPANDED,
                torch.Size([
                    batch_size, num_particles, batch_size, num_particles
                ])
            ],
            [
                state.DistributionBatchShapeMode.BATCH_EXPANDED,
                torch.Size([batch_size, num_particles, num_particles])
            ],
            [
                state.DistributionBatchShapeMode.FULLY_EXPANDED,
                torch.Size([batch_size, num_particles])
            ]
        ]:
            state.set_batch_shape_mode(distribution, batch_shape_mode)
            self.assertEqual(
                state.sample(distribution, batch_size, num_particles).size(),
                sample_size
            )

    def test_sample_values(self):
        for batch_size, num_particles in [(2, 2), (2, 3)]:
            batch_shape = (batch_size, num_particles)
            loc = 100 * torch.arange(batch_size * num_particles).view(
                batch_size, num_particles
            )
            scale = torch.ones(size=batch_shape)
            distribution = torch.distributions.Normal(loc=loc, scale=scale)
            for batch_shape_mode, expanded_dimensions in [
                [
                    state.DistributionBatchShapeMode.NOT_EXPANDED,
                    # batch_size, num_particles, batch_size, num_particles
                    (0, 1)
                ],
                [
                    state.DistributionBatchShapeMode.BATCH_EXPANDED,
                    (1,)  # batch_size, num_particles, num_particles
                ],
                [
                    state.DistributionBatchShapeMode.FULLY_EXPANDED,
                    ()  # batch_size, num_particles
                ]
            ]:
                state.set_batch_shape_mode(distribution, batch_shape_mode)
                samples = state.sample(distribution, batch_size, num_particles)
                mean = mean_dims(samples, dims=expanded_dimensions)

                # within 10 standard deviations
                np.testing.assert_allclose(mean.numpy(), loc.numpy(), atol=10)


class TestLogProb(unittest.TestCase):
    def test_dimensions(self):
        categorical_dim = 5
        for batch_size, num_particles in [(2, 2), (2, 3)]:
            for dims in [(), (4,), (4, 5), (2,), (2, 3)]:
                for batch_shape in [
                    (batch_size, num_particles) + dims,
                    (batch_size,) + dims,
                    dims
                ]:
                    value = torch.rand(size=(batch_size, num_particles) + dims)
                    distribution = torch.distributions.Normal(
                        loc=torch.zeros(size=batch_shape),
                        scale=torch.ones(size=batch_shape)
                    )
                    self.assertEqual(
                        state.log_prob(distribution, value).size(),
                        torch.Size([batch_size, num_particles])
                    )

                    # non-empty event_shape
                    value = torch.zeros(
                        size=(batch_size, num_particles) + dims +
                        (categorical_dim,)
                    ).scatter_(
                        -1,
                        torch.zeros(
                            size=(batch_size, num_particles) + dims +
                            (categorical_dim,)
                        ).long(),
                        1
                    )
                    distribution = torch.distributions.OneHotCategorical(
                        probs=torch.ones(size=dims + (categorical_dim,))
                    )
                    self.assertEqual(
                        state.log_prob(distribution, value).size(),
                        torch.Size([batch_size, num_particles])
                    )

    def test_value(self):
        for batch_size, num_particles in [(2, 2), (2, 3)]:
            for dims in [(), (4,), (4, 5), (2,), (2, 3)]:
                for idx, batch_shape in enumerate([
                    (batch_size, num_particles) + dims,
                    (batch_size,) + dims,
                    dims
                ]):
                    value = torch.zeros(
                        size=(batch_size, num_particles) + dims
                    )
                    loc = 10 * torch.arange(int(np.prod(batch_shape))).view(
                        size=batch_shape
                    )
                    scale = 1
                    distribution = torch.distributions.Normal(
                        loc=loc, scale=scale
                    )
                    if idx == 0:
                        expanded_loc = loc
                    elif idx == 1:
                        expanded_loc = loc.unsqueeze(1).expand(
                            size=(batch_size, num_particles) + dims
                        )
                    else:
                        expanded_loc = loc.unsqueeze(0).unsqueeze(0).expand(
                            size=(batch_size, num_particles) + dims
                        )
                    expanded_scale = scale
                    expanded_distribution = torch.distributions.Normal(
                        loc=expanded_loc,
                        scale=expanded_scale
                    )
                    np.testing.assert_allclose(
                        state.log_prob(distribution, value).numpy(),
                        torch.sum(expanded_distribution.log_prob(value).view(
                            batch_size, num_particles, -1
                        ), dim=2).numpy()
                    )


# class TestLogProbNonReparam(unittest.TestCase):
#     def test_dimensions(self):
#         batch_size, num_particles = 1, 2
#         distribution = torch.distributions.Normal(
#             loc=torch.zeros(3, 4), scale=torch.ones(3, 4)
#         )
#         value = torch.rand(batch_size, num_particles, 3, 4)
#         lp = state.log_prob(distribution, value, non_reparam=True)
#         self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))
#
#         batch_size, num_particles = 1, 2
#         distribution = torch.distributions.Normal(
#             loc=torch.zeros(batch_size, num_particles, 3, 4),
#             scale=torch.ones(batch_size, num_particles, 3, 4)
#         )
#         value = torch.rand(batch_size, num_particles, 3, 4)
#         lp = state.log_prob(distribution, value)
#         self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))
#
#         categorical = torch.distributions.Categorical(
#             probs=torch.Tensor([0.2, 0.3, 0.5])
#         )
#         categorical_value = torch.Tensor([
#             [1, 0, 2],
#             [0, 1, 2]
#         ])
#         batch_size, num_particles = categorical_value.size()
#         normal = torch.distributions.Normal(
#             loc=torch.zeros(batch_size, num_particles, 3, 4),
#             scale=torch.ones(batch_size, num_particles, 3, 4)
#         )
#         normal_value = torch.rand(batch_size, num_particles, 3, 4)
#         distribution = {'categorical': categorical, 'normal': normal}
#         value = {'categorical': categorical_value, 'normal': normal_value}
#         lp = state.log_prob(distribution, value, non_reparam=True)
#         self.assertEqual(lp.size(), torch.Size([batch_size, num_particles]))
#
#     def test_value(self):
#         batch_size, num_particles = 1, 2
#         distribution = torch.distributions.Normal(
#             loc=torch.zeros(3, 4), scale=torch.ones(3, 4)
#         )
#         value = torch.ones(batch_size, num_particles, 3, 4)
#         lp = state.log_prob(distribution, value, non_reparam=True)
#         np.testing.assert_equal(
#             lp.numpy(), np.zeros([batch_size, num_particles])
#         )
#
#         categorical = torch.distributions.Categorical(
#             probs=torch.Tensor([0.2, 0.3, 0.5])
#         )
#         categorical_value = torch.Tensor([
#             [1, 0, 2],
#             [0, 1, 2]
#         ])
#         batch_size, num_particles = categorical_value.size()
#         normal = torch.distributions.Normal(
#             loc=torch.zeros(batch_size, num_particles, 3, 4),
#             scale=torch.ones(batch_size, num_particles, 3, 4)
#         )
#         normal_value = torch.rand(batch_size, num_particles, 3, 4)
#         distribution = {'categorical': categorical, 'normal': normal}
#         value = {'categorical': categorical_value, 'normal': normal_value}
#         lp = state.log_prob(distribution, value, non_reparam=True)
#         np.testing.assert_equal(
#             lp.numpy(), state.log_prob(categorical, categorical_value).numpy()
#         )
#
#
# class TestResample(unittest.TestCase):
#     def test_dimensions(self):
#         ancestral_index = torch.zeros(3, 2).long()
#         value = torch.rand(3, 2)
#         self.assertEqual(
#             value.size(),
#             state.resample(value, ancestral_index).size()
#         )
#
#         value = torch.rand(3, 2, 4, 5)
#         self.assertEqual(
#             value.size(),
#             state.resample(value, ancestral_index).size()
#         )
#
#     def test_small(self):
#         ancestral_index = torch.LongTensor([
#             [1, 2, 0],
#             [0, 0, 1]
#         ])
#         value = torch.Tensor([
#             [1, 2, 3],
#             [4, 5, 6]
#         ])
#         resampled_value = torch.Tensor([
#             [2, 3, 1],
#             [4, 4, 5]
#         ])
#
#         self.assertTrue(torch.equal(
#             state.resample(value, ancestral_index),
#             resampled_value
#         ))


class TestExpandObservation(unittest.TestCase):
    def test_dimensions(self):
        batch_size, num_particles = 2, 3
        dims_list = [(), (4,), (4, 5)]
        for dims in dims_list:
            observation = torch.rand(size=(batch_size,) + dims)
            self.assertEqual(
                state.expand_observation(observation, num_particles).size(),
                torch.Size((batch_size, num_particles) + dims)
            )

        for a_dims in dims_list:
            for b_dims in dims_list:
                observation = {
                    'a': torch.rand(size=(batch_size,) + a_dims),
                    'b': torch.rand(size=(batch_size,) + b_dims)
                }
                self.assertEqual(
                    state.expand_observation(
                        observation, num_particles
                    )['a'].size(),
                    torch.Size((batch_size, num_particles) + a_dims)
                )
                self.assertEqual(
                    state.expand_observation(
                        observation, num_particles
                    )['b'].size(),
                    torch.Size((batch_size, num_particles) + b_dims)
                )


if __name__ == '__main__':
    unittest.main()

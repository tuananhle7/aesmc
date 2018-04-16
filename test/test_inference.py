import dgm.inference as inference
import dgm.model as model
import dgm.state as state
import dgm.statistics as stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch.nn as nn
import pykalman
import torch
import unittest


class TestGetResampledLatentStates(unittest.TestCase):
    def test_value(self):
        return
        latents = [
            torch.Tensor([[1, 2, 3]]),
            torch.Tensor([[4, 5, 6]]),
            torch.Tensor([[7, 8, 9]]),
            torch.Tensor([[10, 11, 12]])
        ]
        ancestral_indices = [
            torch.LongTensor([[0, 2, 1]]),
            torch.LongTensor([[2, 0, 0]]),
            torch.LongTensor([[1, 2, 0]])
        ]
        true_resampled_latents = [
            torch.Tensor([[1, 1, 2]]),
            torch.Tensor([[4, 4, 6]]),
            torch.Tensor([[8, 9, 7]]),
            torch.Tensor([[10, 11, 12]])
        ]

        resampled_latents = inference.get_resampled_latents(
            latents, ancestral_indices
        )
        for i, resampled_latent in enumerate(resampled_latents):
            np.testing.assert_equal(
                resampled_latent[0].numpy(),
                true_resampled_latents[i][0].numpy()
            )


class TestSampleAncestralIndex(unittest.TestCase):
    def test_dimensions(self):
        return
        self.assertEqual(
            inference.sample_ancestral_index(torch.rand(2, 3)).size(),
            torch.Size([2, 3])
        )
        self.assertEqual(
            inference.sample_ancestral_index(torch.rand(1, 2)).size(),
            torch.Size([1, 2])
        )
        self.assertEqual(
            inference.sample_ancestral_index(torch.rand(2, 1)).size(),
            torch.Size([2, 1])
        )

    def test_type(self):
        return
        self.assertIsInstance(
            inference.sample_ancestral_index(torch.rand(1, 1)),
            torch.LongTensor
        )

    def test_sampler(self):
        weight = [0.2, 0.3, 0.5]
        num_trials = 10000
        ancestral_indices = inference.sample_ancestral_index(
            torch.log(torch.Tensor(weight)).unsqueeze(0).expand(
                num_trials, len(weight)
            )
        )

        empirical_probabilities = []
        for i in range(len(weight)):
            empirical_probabilities.append(
                torch.sum((ancestral_indices == i).float()).item() /
                (num_trials * len(weight))
            )

        np.testing.assert_allclose(
            np.array(empirical_probabilities),
            np.array(weight),
            atol=1e-2  # 2 decimal places
        )


class MyInitialDistribution(model.InitialDistribution):
    def __init__(self, initial_mean, initial_variance):
        self.initial_mean = initial_mean
        self.initial_variance = initial_variance

    def initial(self):
        return torch.distributions.Normal(
            loc=self.initial_mean,
            scale=np.sqrt(self.initial_variance)
        )


class MyTransitionDistribution(model.TransitionDistribution):
    def __init__(
        self, transition_matrix, transition_covariance, transition_offset
    ):
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.transition_offset = transition_offset

    def transition(self, previous_latent=None, time=None):
        return torch.distributions.Normal(
            loc=previous_latent * self.transition_matrix +
            self.transition_offset,
            scale=np.sqrt(self.transition_covariance)
        )


class MyEmissionDistribution(model.EmissionDistribution):
    def __init__(
        self, emission_matrix, emission_covariance, emission_offset
    ):
        self.emission_matrix = emission_matrix
        self.emission_covariance = emission_covariance
        self.emission_offset = emission_offset

    def emission(self, latent=None, time=None):
        return torch.distributions.Normal(
            loc=latent * self.emission_matrix +
            self.emission_offset,
            scale=np.sqrt(self.emission_covariance)
        )


class MyProposalDistribution(model.ProposalDistribution):
    def __init__(
        self, initial_mean, initial_variance, transition_matrix,
        transition_covariance, transition_offset
    ):
        self.initial_mean = initial_mean
        self.initial_variance = initial_variance
        self.transition_matrix = transition_matrix
        self.transition_covariance = transition_covariance
        self.transition_offset = transition_offset

    def proposal(
        self, previous_latent=None, time=None, observations=None
    ):
        if time == 0:
            return torch.distributions.Normal(
                loc=self.initial_mean,
                scale=np.sqrt(self.initial_variance)
            )
        else:
            return torch.distributions.Normal(
                loc=previous_latent * self.transition_matrix +
                self.transition_offset,
                scale=np.sqrt(self.transition_covariance)
            )


class TestInfer(unittest.TestCase):
    # Test inference against a Kalman filter. Outputs plots to
    # test/test_inference_plots/

    @classmethod
    def setUpClass(self):
        super(TestInfer, self).setUpClass()

        # Synthetic data
        self.num_timesteps = 100
        self.x = np.linspace(0, 3 * np.pi, self.num_timesteps)
        self.observations = 40 * (
            np.sin(self.x) + 0.2 * np.random.randn(self.num_timesteps)
        )

        kf = pykalman.KalmanFilter(
            transition_matrices=[[1]],
            transition_covariance=0.01 * np.eye(1),
        )

        # EM to fit parameters
        kf = kf.em(self.observations)

        # Inference using Kalman filter
        self.kalman_smoothed_state_means, \
            self.kalman_smoothed_state_variances = kf.smooth(self.observations)

        # Prepare state space model
        # self.batch_size = 1
        self.num_particles = 1000
        self.observations_tensor = torch.from_numpy(self.observations).\
            unsqueeze(-1).float()

        self.my_initial_distribution = MyInitialDistribution(
            initial_mean=float(kf.initial_state_mean[0]),
            initial_variance=float(kf.initial_state_covariance[0][0])
        )
        self.my_transition_distribution = MyTransitionDistribution(
            transition_matrix=float(kf.transition_matrices[0][0]),
            transition_covariance=float(kf.transition_covariance[0][0]),
            transition_offset=float(kf.transition_offsets[0])
        )
        self.my_emission_distribution = MyEmissionDistribution(
            emission_matrix=float(kf.observation_matrices[0][0]),
            emission_covariance=float(kf.observation_covariance[0][0]),
            emission_offset=float(kf.observation_offsets[0])
        )
        self.my_proposal_distribution = MyProposalDistribution(
            initial_mean=float(kf.initial_state_mean[0]),
            initial_variance=float(kf.initial_state_covariance[0][0]),
            transition_matrix=float(kf.transition_matrices[0][0]),
            transition_covariance=float(kf.transition_covariance[0][0]),
            transition_offset=float(kf.transition_offsets[0])
        )

    def test_importance_sampling(self):
        return
        # Inference using importance sampling
        inference_result = inference.infer(
            inference_algorithm=inference.InferenceAlgorithm.IS,
            observations=self.observations_tensor,
            initial=self.my_initial_distribution,
            transition=self.my_transition_distribution,
            emission=self.my_emission_distribution,
            proposal=self.my_proposal_distribution,
            num_particles=self.num_particles
        )

        is_smoothed_state_means = []
        is_smoothed_state_variances = []
        for latent in inference_result['latents']:
            is_smoothed_state_means.append(stats.empirical_mean(
                latent, inference_result['log_weight']
            )[0])
            is_smoothed_state_variances.append(stats.empirical_variance(
                latent, inference_result['log_weight']
            )[0])

        # Plotting
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(8, 6)
        fig.suptitle(
            'Importance sampling with {} particles'.format(self.num_particles)
        )
        ax.plot(
            self.x,
            self.observations,
            label='observations',
            linewidth=1,
            color='black'
        )
        kalman_line = ax.plot(
            self.x,
            self.kalman_smoothed_state_means[:, 0],
            linewidth=1,
            label='kalman'
        )
        ax.fill_between(
            self.x,
            self.kalman_smoothed_state_means[:, 0] -
            np.sqrt(self.kalman_smoothed_state_variances[:, 0, 0]),
            self.kalman_smoothed_state_means[:, 0] +
            np.sqrt(self.kalman_smoothed_state_variances[:, 0, 0]),
            alpha=0.2,
            color=kalman_line[0].get_color()
        )
        is_line = ax.plot(
            self.x,
            np.array(is_smoothed_state_means),
            linewidth=1,
            linestyle='dashed',
            label='importance sampling'
        )
        ax.fill_between(
            self.x,
            np.array(is_smoothed_state_means) -
            np.sqrt(np.array(is_smoothed_state_variances)),
            np.array(is_smoothed_state_means) +
            np.sqrt(np.array(is_smoothed_state_variances)),
            alpha=0.2,
            color=is_line[0].get_color()
        )
        ax.legend()
        ax.set_xlim([self.x[0], self.x[-1]])
        ax.set_xlabel('time')
        ax.set_ylabel('smoothed means $\pm$ 1 standard deviation')
        filename = './test/test_inference_plots/test_importance_sampling.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

        # Tests
        mean_sqmse = np.sqrt(np.mean(
            (self.kalman_smoothed_state_means[:, 0] -
                np.array(is_smoothed_state_means))**2
        ))
        variance_avg_relative_error = np.mean(
            np.abs(
                self.kalman_smoothed_state_variances[:, 0, 0] -
                np.array(is_smoothed_state_variances)
            ) / self.kalman_smoothed_state_variances[:, 0, 0]
        )
        # We expect importance sampling to perform very badly
        self.assertLess(mean_sqmse, 20)
        self.assertLessEqual(variance_avg_relative_error, 2)

    def test_smc(self):
        # Inference using SMC
        return
        inference_result = inference.infer(
            inference_algorithm=inference.InferenceAlgorithm.SMC,
            observations=self.observations_tensor,
            initial=self.my_initial_distribution,
            transition=self.my_transition_distribution,
            emission=self.my_emission_distribution,
            proposal=self.my_proposal_distribution,
            num_particles=self.num_particles
        )

        smc_smoothed_state_means = []
        smc_smoothed_state_variances = []
        for latent in inference_result['latents']:
            smc_smoothed_state_means.append(stats.empirical_mean(
                latent, inference_result['log_weight']
            )[0])
            smc_smoothed_state_variances.append(stats.empirical_variance(
                latent, inference_result['log_weight']
            )[0])

        # Plotting
        fig, ax = plt.subplots(nrows=1, ncols=1)
        fig.set_size_inches(8, 6)
        fig.suptitle('SMC with {} particles'.format(self.num_particles))
        ax.plot(
            self.x,
            self.observations,
            label='observations',
            linewidth=1,
            color='black'
        )
        kalman_line = ax.plot(
            self.x,
            self.kalman_smoothed_state_means[:, 0],
            linewidth=1,
            label='kalman'
        )
        ax.fill_between(
            self.x,
            self.kalman_smoothed_state_means[:, 0] -
            np.sqrt(self.kalman_smoothed_state_variances[:, 0, 0]),
            self.kalman_smoothed_state_means[:, 0] +
            np.sqrt(self.kalman_smoothed_state_variances[:, 0, 0]),
            alpha=0.2,
            color=kalman_line[0].get_color()
        )
        smc_line = ax.plot(
            self.x,
            np.array(smc_smoothed_state_means),
            linewidth=1,
            linestyle='dotted',
            label='smc'
        )
        ax.fill_between(
            self.x,
            np.array(smc_smoothed_state_means) -
            np.sqrt(np.array(smc_smoothed_state_variances)),
            np.array(smc_smoothed_state_means) +
            np.sqrt(np.array(smc_smoothed_state_variances)),
            alpha=0.2,
            color=smc_line[0].get_color()
        )
        ax.legend()
        ax.set_xlim([self.x[0], self.x[-1]])
        ax.set_xlabel('time')
        ax.set_ylabel('smoothed means $\pm$ 1 standard deviation')
        filename = './test/test_inference_plots/test_smc.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

        # Tests
        mean_sqmse = np.sqrt(np.mean(
            (self.kalman_smoothed_state_means[:, 0] -
                np.array(smc_smoothed_state_means))**2
        ))
        variance_avg_relative_error = np.mean(
            np.abs(
                self.kalman_smoothed_state_variances[:, 0, 0] -
                np.array(smc_smoothed_state_variances)
            ) / self.kalman_smoothed_state_variances[:, 0, 0]
        )
        # We expect SMC to perform well
        self.assertLess(mean_sqmse, 2)
        self.assertLess(variance_avg_relative_error, 0.5)


class TestAncestralIndicesLogProb(unittest.TestCase):
    def test_dimensions(self):
        return
        batch_size = 3
        num_particles = 4
        self.assertEqual(
            inference.ancestral_indices_log_prob(
                [torch.ones(batch_size, num_particles).long()],
                [torch.rand(batch_size, num_particles),
                 torch.rand(batch_size, num_particles)]
            ).size(),
            torch.Size([batch_size])
        )
        self.assertEqual(
            inference.ancestral_indices_log_prob(
                [], [torch.rand(batch_size, num_particles)]
            ).size(),
            torch.Size([batch_size])
        )

    def test_type(self):
        return
        batch_size = 1
        num_particles = 1
        self.assertIsInstance(
            inference.ancestral_indices_log_prob(
                [torch.zeros(batch_size, num_particles).long()],
                [torch.rand(batch_size, num_particles),
                 torch.rand(batch_size, num_particles)]
            ),
            torch.Tensor
        )
        self.assertIsInstance(
            inference.ancestral_indices_log_prob(
                [], [torch.rand(batch_size, num_particles)]
            ),
            torch.Tensor
        )

    def test_value(self):
        return
        weights = [[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]
        ancestral_indices = [[1, 0], [0, 0]]
        self.assertAlmostEqual(
            inference.ancestral_indices_log_prob(
                list(map(
                    lambda ancestral_index:
                        torch.Tensor(ancestral_index).long().unsqueeze(0),
                    ancestral_indices
                )),
                list(map(
                    lambda weight:
                        torch.log(
                            torch.Tensor(weight) * np.random.rand()
                        ).unsqueeze(0),
                    weights
                ))
            ).item(),
            np.log(0.8 * 0.2 * 0.6 * 0.6),
            places=6  # Fails with 7 places
        )


class MyProposalNetwork(model.ProposalNetwork):
    def __init__(self, proposal_multiplier):
        super(MyProposalNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([proposal_multiplier]))
        self.std = 1

    def proposal(
        self, previous_latent=None, time=None, observations=None
    ):
        if time == 0:
            return torch.distributions.Normal(loc=0, scale=1)
        else:
            return torch.distributions.Normal(
                loc=self.multiplier * previous_latent,
                scale=self.std
            )


class TestLatentsLogProb(unittest.TestCase):
    def test_dimensions(self):
        return
        for num_timesteps, batch_size, num_particles in [
            (2, 3, 4), (1, 1, 1), (2, 1, 1)
        ]:
            my_proposal_network = MyProposalNetwork(1)
            original_latents = list(
                torch.rand(num_timesteps, batch_size, num_particles)
            )
            log_weights = list(
                torch.rand(num_timesteps, batch_size, num_particles)
            )
            ancestral_indices = list(
                torch.rand(num_timesteps - 1, batch_size, num_particles).long()
            )

            for non_reparam in [True, False]:
                self.assertEqual(
                    inference.latents_log_prob(
                        my_proposal_network,
                        None,
                        original_latents,
                        ancestral_indices,
                        non_reparam=non_reparam
                    ).size(),
                    torch.Size([batch_size])
                )
                self.assertEqual(
                    inference.latents_log_prob(
                        my_proposal_network,
                        None,
                        original_latents,
                        non_reparam=non_reparam
                    ).size(),
                    torch.Size([batch_size])
                )

    def test_value(self):
        return
        batch_size, num_particles, num_timesteps = (1, 2, 3)
        my_proposal_network = MyProposalNetwork(1)
        log_weights = list(torch.log(
            torch.Tensor([[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]).unsqueeze(1)
        ))
        ancestral_indices = list(
            torch.Tensor([[1, 0], [0, 0]]).unsqueeze(1).long()
        )
        original_latents = list(
            torch.Tensor([[1, 2], [3, 4], [5, 6]]).unsqueeze(1)
        )

        self.assertAlmostEqual(
            inference.latents_log_prob(
                my_proposal_network,
                None,
                original_latents,
                ancestral_indices
            ).item(),
            scipy.stats.norm.logpdf(1, 0, 1) +
            scipy.stats.norm.logpdf(2, 0, 1) +
            scipy.stats.norm.logpdf(3, 2, 1) +
            scipy.stats.norm.logpdf(4, 1, 1) +
            scipy.stats.norm.logpdf(5, 3, 1) +
            scipy.stats.norm.logpdf(6, 3, 1),
            places=5
        )
        self.assertAlmostEqual(
            inference.latents_log_prob(
                my_proposal_network,
                None,
                original_latents
            ).item(),
            scipy.stats.norm.logpdf(1, 0, 1) +
            scipy.stats.norm.logpdf(2, 0, 1) +
            scipy.stats.norm.logpdf(3, 1, 1) +
            scipy.stats.norm.logpdf(4, 2, 1) +
            scipy.stats.norm.logpdf(5, 3, 1) +
            scipy.stats.norm.logpdf(6, 4, 1),
            places=5
        )
        self.assertAlmostEqual(
            inference.latents_log_prob(
                my_proposal_network,
                None,
                original_latents,
                ancestral_indices,
                non_reparam=True
            ).item(),
            0,
            places=5
        )
        self.assertAlmostEqual(
            inference.latents_log_prob(
                my_proposal_network,
                None,
                original_latents,
                non_reparam=True
            ).item(),
            0,
            places=5
        )
        # TODO: test for actual non-reparameterizable latents

if __name__ == '__main__':
    pass
    #  unittest.main()

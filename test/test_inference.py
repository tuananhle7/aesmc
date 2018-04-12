import dgm.inference as inference
import dgm.model as model
import dgm.state as state
import dgm.statistics as stats
import matplotlib.pyplot as plt
import numpy as np
import pykalman
import torch
import unittest


class TestGetResampledLatentStates(unittest.TestCase):
    def test_value(self):
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
    # test/test_inference/

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
        # Inference using importance sampling
        inference_result = inference.infer(
            algorithm='is',
            observations=self.observations_tensor,
            initial=self.my_initial_distribution,
            transition=self.my_transition_distribution,
            emission=self.my_emission_distribution,
            proposal=self.my_proposal_distribution,
            num_particles=self.num_particles,
            reparameterized=False
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
        self.assertLessEqual(variance_avg_relative_error, 1)

    def test_smc(self):
        # Inference using SMC
        inference_result = inference.infer(
            algorithm='smc',
            observations=self.observations_tensor,
            initial=self.my_initial_distribution,
            transition=self.my_transition_distribution,
            emission=self.my_emission_distribution,
            proposal=self.my_proposal_distribution,
            num_particles=self.num_particles,
            reparameterized=False
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
        self.assertLess(variance_avg_relative_error, 0.3)


if __name__ == '__main__':
    unittest.main()

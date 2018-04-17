import dgm
import dgm.autoencoder as ae
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import unittest


class MyInitialNetwork(dgm.model.InitialNetwork):
    def __init__(self, initial_mean):
        super(MyInitialNetwork, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([initial_mean]))
        self.std = 1

    def initial(self):
        return torch.distributions.Normal(loc=self.mean, scale=self.std)


class MyTransitionNetwork(dgm.model.TransitionNetwork):
    def __init__(self, transition_multiplier):
        super(MyTransitionNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([transition_multiplier]))
        self.std = 1

    def transition(self, previous_latent=None, time=None):
        return torch.distributions.Normal(
            loc=self.multiplier * previous_latent,
            scale=self.std
        )


class MyEmissionNetwork(dgm.model.EmissionNetwork):
    def __init__(self, emission_multiplier):
        super(MyEmissionNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([emission_multiplier]))
        self.std = 1

    def emission(self, latent=None, time=None):
        return torch.distributions.Normal(
            loc=self.multiplier * latent,
            scale=self.std
        )


class MyProposalNetwork(dgm.model.ProposalNetwork):
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


class TestAutoEncoder(unittest.TestCase):
    def test_dimensions(self):
        batch_size = 4
        num_particles = 5
        num_timesteps = 6
        observations = list(torch.rand(num_timesteps, batch_size))

        my_initial_network = MyInitialNetwork(0)
        my_transition_network = MyTransitionNetwork(1.2)
        my_emission_network = MyEmissionNetwork(0.9)
        my_proposal_network = MyProposalNetwork(1.1)
        my_auto_encoder = ae.AutoEncoder(
            initial=my_initial_network,
            transition=my_transition_network,
            emission=my_emission_network,
            proposal=my_proposal_network,
        )

        for resampling_gradient_estimator in ae.ResamplingGradientEstimator:
            elbo = my_auto_encoder.forward(
                observations=observations,
                num_particles=num_particles,
                autoencoder_algorithm=ae.AutoencoderAlgorithm.AESMC,
                resampling_gradient_estimator=resampling_gradient_estimator
            )
            self.assertEqual(elbo.size(), torch.Size([batch_size]))
            torch.mean(elbo).backward()
            self.assertEqual(
                my_initial_network.mean.size(),
                my_initial_network.mean.grad.size()
            )

    def test_gaussian(self):
        from .models import gaussian

        prior_std = 1

        true_prior_mean = 0
        true_obs_std = 1

        prior_mean_init = 2
        obs_std_init = 0.5

        q_init_mult, q_init_bias, q_init_std = 2, 2, 2
        q_true_mult, q_true_bias, q_true_std = gaussian.get_proposal_params(
            true_prior_mean, prior_std, true_obs_std
        )

        true_prior = gaussian.Prior(true_prior_mean, prior_std)
        true_likelihood = gaussian.Likelihood(true_obs_std)

        prior = gaussian.Prior(prior_mean_init, prior_std)
        likelihood = gaussian.Likelihood(obs_std_init)
        inference_network = gaussian.InferenceNetwork(
            q_init_mult, q_init_bias, q_init_std
        )

        autoencoder = ae.AutoEncoder(
            prior, None, likelihood, inference_network
        )

        num_particles = 2
        batch_size = 10
        num_iterations = 2000

        training_stats = gaussian.TrainingStats(logging_interval=500)

        print('\nTraining the \"gaussian\" autoencoder.')
        dgm.train.train_autoencoder(
            autoencoder,
            dgm.train.get_synthetic_dataloader(
                true_prior, None, true_likelihood, 1, batch_size
            ),
            autoencoder_algorithm=dgm.autoencoder.AutoencoderAlgorithm.IWAE,
            #  autoencoder_algorithm=dgm.autoencoder.AutoencoderAlgorithm.WAKE_SLEEP,
            num_epochs=1,
            num_iterations_per_epoch=num_iterations,
            num_particles=num_particles,
            wake_sleep_mode=ae.WakeSleepAlgorithm.WW,
            optimizer_algorithm=torch.optim.SGD,
            optimizer_kwargs={'lr': 0.01},
            #  discrete_gradient_estimator=ae.DiscreteGradientEstimator.VIMCO,
            callback=training_stats
        )

        fig, axs = plt.subplots(5, 1, sharex=True)
        fig.set_size_inches(5, 8)

        for ax, data, true, ylabel in zip(
            axs,
            [
                training_stats.prior_mean_history,
                training_stats.obs_std_history,
                training_stats.q_mult_history,
                training_stats.q_bias_history,
                training_stats.q_std_history
            ],
            [true_prior_mean, true_obs_std, q_true_mult,
             q_true_bias, q_true_std],
            ['$\mu_0$', '$\sigma$', '$a$', '$b$', '$c$']
        ):
            ax.plot(training_stats.iteration_idx_history, data)
            ax.axhline(true, color='black')
            ax.set_ylabel(ylabel)
            #  self.assertAlmostEqual(data[-1], true, delta=1e-1)

        axs[-1].set_xlabel('Iteration')
        axs[0].set_title('Gaussian')
        fig.tight_layout()

        filename = './test/test_autoencoder_plots/gaussian.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

    def test_gmm(self):
        from .models import gmm
        num_mixtures = 5

        temp = np.arange(num_mixtures) + 5
        true_mixture_probs = temp / np.sum(temp)
        init_mixture_probs_pre_softmax = np.array(
            list(reversed(2 * np.arange(num_mixtures)))
        )

        mean_multiplier = 10

        stds = np.array([5 for _ in range(num_mixtures)])
        softmax_multiplier = 0.5

        true_prior = gmm.Prior(
            init_mixture_probs_pre_softmax=np.log(
                true_mixture_probs
            ) / softmax_multiplier,
            softmax_multiplier=softmax_multiplier
        )
        likelihood = gmm.Likelihood(mean_multiplier, stds)
        prior = gmm.Prior(
            init_mixture_probs_pre_softmax=init_mixture_probs_pre_softmax,
            softmax_multiplier=softmax_multiplier
        )
        inference_network = gmm.InferenceNetwork(num_mixtures)

        autoencoder = dgm.autoencoder.AutoEncoder(
            prior, None, likelihood, inference_network
        )

        num_particles = 5
        batch_size = 100
        num_iterations = 20000

        training_stats = gmm.TrainingStats(logging_interval=1000)
        dataloader = dgm.train.get_synthetic_dataloader(
            true_prior, None, likelihood, 1, batch_size
        )

        dgm.train.train_autoencoder(
            autoencoder,
            dataloader,
            autoencoder_algorithm=dgm.autoencoder.AutoencoderAlgorithm.IWAE,
            num_epochs=1,
            num_iterations_per_epoch=num_iterations,
            num_particles=num_particles,
            #  wake_sleep_mode=ae.WakeSleepAlgorithm.WW,
            discrete_gradient_estimator=ae.DiscreteGradientEstimator.VIMCO,
            callback=training_stats
        )

        num_test_data = 100
        prior_l2, posterior_l2 = gmm.get_stats(
            training_stats.mixture_probs_history,
            training_stats.inference_network_state_dict_history,
            true_prior, likelihood, num_test_data
        )

        # Plotting
        fig, axs = plt.subplots(2, 1, sharex=True)
        fig.set_size_inches(4, 4)

        for idx, (ax, data, ylabel) in enumerate(zip(
            axs,
            [prior_l2, posterior_l2],
            ['$|| p_{\\theta}(z) - p_{\\theta^*}(z) ||$',
             'Avg. test\n$|| q_\phi(z | x) - p_{\\theta^*}(z | x) ||$']
        )):
            ax.plot(training_stats.iteration_idx_history, data)
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axs[0].set_title('Gaussian Mixture Model')
        axs[-1].set_xlabel('Iteration')
        fig.tight_layout()
        filename = './test/test_autoencoder_plots/gmm.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

        self.assertAlmostEqual(prior_l2[-1], 0, delta=1e-1)
        self.assertAlmostEqual(posterior_l2[-1], 0, delta=6e-1)


if __name__ == '__main__':
    unittest.main()

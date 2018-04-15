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
        num_iterations = 5000

        training_stats = gaussian.TrainingStats(logging_interval=500)

        print('\n---\nTraining the \"gaussian\" autoencoder.')
        dgm.train.train_autoencoder(
            autoencoder,
            dgm.train.get_synthetic_dataloader(
                true_prior, None, true_likelihood, 1, batch_size
            ),
            autoencoder_algorithm=dgm.autoencoder.AutoencoderAlgorithm.IWAE,
            num_epochs=1,
            num_iterations_per_epoch=num_iterations,
            num_particles=num_particles,
            optimizer_algorithm=torch.optim.SGD,
            optimizer_kwargs={'lr': 0.01},
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
            self.assertAlmostEqual(data[-1], true, delta=1e-1)

        axs[-1].set_xlabel('Iteration')
        axs[0].set_title('Gaussian')
        fig.tight_layout()

        filename = './test/test_autoencoder_plots/gaussian.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))


if __name__ == '__main__':
    unittest.main()

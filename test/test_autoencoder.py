import dgm
import dgm.autoencoder as ae
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import unittest


class MyInitialNetwork(nn.Module):
    def __init__(self, initial_mean):
        super(MyInitialNetwork, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([initial_mean]))
        self.std = 1

    def __call__(self):
        return torch.distributions.Normal(loc=self.mean, scale=self.std)


class MyTransitionNetwork(nn.Module):
    def __init__(self, transition_multiplier):
        super(MyTransitionNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([transition_multiplier]))
        self.std = 1

    def __call__(self, previous_latent=None, time=None):
        return torch.distributions.Normal(
            loc=self.multiplier * previous_latent, scale=self.std)


class MyEmissionNetwork(nn.Module):
    def __init__(self, emission_multiplier):
        super(MyEmissionNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([emission_multiplier]))
        self.std = 1

    def __call__(self, latent=None, time=None):
        return torch.distributions.Normal(loc=self.multiplier * latent,
                                          scale=self.std)


class MyProposalNetwork(nn.Module):
    def __init__(self, proposal_multiplier):
        super(MyProposalNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([proposal_multiplier]))
        self.std = 1

    def forward(self, previous_latent=None, time=None, observations=None):
        if time == 0:
            return torch.distributions.Normal(loc=0, scale=1)
        else:
            return torch.distributions.Normal(
                loc=self.multiplier * previous_latent, scale=self.std)


class MeanStdAccum():
    def __init__(self):
        self.count = 0
        self.means = None
        self.M2s = None

    def update(self, new_variables):
        if self.count == 0:
            self.count = 1
            self.means = []
            self.M2s = []
            for new_var in new_variables:
                self.means.append(new_var.data)
                self.M2s.append(new_var.data.new(new_var.size()).fill_(0))
        else:
            self.count = self.count + 1
            for new_var_idx, new_var in enumerate(new_variables):
                delta = new_var.data - self.means[new_var_idx]
                self.means[new_var_idx] = self.means[new_var_idx] + delta / self.count
                delta_2 = new_var.data - self.means[new_var_idx]
                self.M2s[new_var_idx] = self.M2s[new_var_idx] + delta * delta_2

    def means_stds(self):
        if self.count < 2:
            raise ArithmeticError('Need more than 1 value. Have {}'.format(self.count))
        else:
            stds = []
            for i in range(len(self.means)):
                stds.append(torch.sqrt(self.M2s[i] / self.count))
            return self.means, stds

    def avg_of_means_stds(self):
        means, stds = self.means_stds()
        num_parameters = np.sum([len(p) for p in means])
        return (
            np.sum([torch.sum(p) for p in means]) / num_parameters,
            np.sum([torch.sum(p) for p in stds]) / num_parameters
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

        elbo = my_auto_encoder.forward(
            observations=observations,
            num_particles=num_particles,
            autoencoder_algorithm=ae.AutoencoderAlgorithm.AESMC,
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
            true_prior_mean, prior_std, true_obs_std)

        true_prior = gaussian.Prior(true_prior_mean, prior_std)
        true_likelihood = gaussian.Likelihood(true_obs_std)

        num_particles = 2
        batch_size = 10
        num_iterations = 2000

        training_stats = gaussian.TrainingStats(logging_interval=500)

        print('\nTraining the \"gaussian\" autoencoder.')
        prior = gaussian.Prior(prior_mean_init, prior_std)
        likelihood = gaussian.Likelihood(obs_std_init)
        inference_network = gaussian.InferenceNetwork(
            q_init_mult, q_init_bias, q_init_std)
        autoencoder = ae.AutoEncoder(
            prior, None, likelihood, inference_network)
        dgm.train.train_autoencoder(
            autoencoder,
            dgm.train.get_synthetic_dataloader(
                true_prior, None, true_likelihood, 1, batch_size),
            autoencoder_algorithm=ae.AutoencoderAlgorithm.IWAE,
            num_epochs=1,
            num_iterations_per_epoch=num_iterations,
            num_particles=num_particles,
            optimizer_algorithm=torch.optim.SGD,
            optimizer_kwargs={'lr': 0.01},
            callback=training_stats
        )

        fig, axs = plt.subplots(5, 1, sharex=True, sharey=True)
        fig.set_size_inches(10, 8)

        mean = training_stats.prior_mean_history
        obs = training_stats.obs_std_history
        mult = training_stats.q_mult_history
        bias = training_stats.q_bias_history
        std = training_stats.q_std_history
        data = [mean] + [obs] + [mult] + [bias] + [std]
        true = [true_prior_mean, true_obs_std, q_true_mult, q_true_bias,
                q_true_std]

        for ax, data_, true_, ylabel in zip(
            axs, data, true, ['$\mu_0$', '$\sigma$', '$a$', '$b$', '$c$']):
            ax.plot(training_stats.iteration_idx_history, data_)
            ax.axhline(true_, color='black')
            ax.set_ylabel(ylabel)
            #  self.assertAlmostEqual(data[-1], true, delta=1e-1)


        axs[-1].set_xlabel('Iteration')
        fig.tight_layout()

        filename = './test/test_autoencoder_plots/gaussian.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

    def test_lgssm(self):
        from .models import lgssm
        print('\nTraining the \"linear Gaussian state space model\"'
              ' autoencoder.')
        initial_loc = 0
        initial_scale = 1
        true_transition_mult = 0.9
        init_transition_mult = 0
        transition_scale = 1
        true_emission_mult = 1
        init_emission_mult = 0
        emission_scale = 0.01
        num_timesteps = 200
        num_test_obs = 10
        test_inference_num_particles = 100
        saving_interval = 10
        logging_interval = 10
        batch_size = 10
        num_iterations = 20 # 500
        num_particles = 100

        # http://tuananhle.co.uk/notes/optimal-proposal-lgssm.html
        optimal_proposal_scale_0 = np.sqrt(
            initial_scale**2 - initial_scale**2 * true_emission_mult /
            (emission_scale**2 + initial_scale**2 * true_emission_mult**2) *
            true_emission_mult * initial_scale**2
        )
        optimal_proposal_scale_t = np.sqrt(
            transition_scale**2 - transition_scale**2 * true_emission_mult /
            (emission_scale**2 + transition_scale**2 * true_emission_mult**2)
            * true_emission_mult * transition_scale**2
        )
        autoencoder_algorithms = [dgm.autoencoder.AutoencoderAlgorithm.IWAE,
                                  dgm.autoencoder.AutoencoderAlgorithm.AESMC]
        names = ['IWAE', 'AESMC']
        dataloader = dgm.train.get_synthetic_dataloader(
            lgssm.Initial(initial_loc, initial_scale),
            lgssm.Transition(true_transition_mult, transition_scale),
            lgssm.Emission(true_emission_mult, emission_scale),
            num_timesteps, batch_size
        )
        fig, axs = plt.subplots(2, 1, sharex=True)
        for name, autoencoder_algorithm in zip(names, autoencoder_algorithms):
            training_stats = lgssm.TrainingStats(
                initial_loc, initial_scale, true_transition_mult,
                transition_scale, true_emission_mult, emission_scale,
                num_timesteps, num_test_obs, test_inference_num_particles,
                saving_interval, logging_interval
            )
            autoencoder = dgm.autoencoder.AutoEncoder(
                lgssm.Initial(initial_loc, initial_scale),
                lgssm.Transition(init_transition_mult, transition_scale),
                lgssm.Emission(init_emission_mult, emission_scale),
                lgssm.Proposal(optimal_proposal_scale_0,
                               optimal_proposal_scale_t)
            )
            dgm.train.train_autoencoder(
                autoencoder, dataloader, autoencoder_algorithm, 1,
                num_iterations, num_particles, callback=training_stats
            )
            axs[0].plot(training_stats.iteration_idx_history,
                        training_stats.p_l2_history,
                        label=name)
            axs[1].plot(training_stats.iteration_idx_history,
                        training_stats.q_l2_history,
                        label=name)
        axs[0].set_ylabel('$||\\theta - \\theta_{true}||$')
        axs[1].set_ylabel('Avg. L2 of\nmarginal posterior means')
        axs[-1].set_xlabel('Iteration')
        axs[0].legend()
        fig.tight_layout()
        filename = './test/test_autoencoder_plots/lgssm.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()

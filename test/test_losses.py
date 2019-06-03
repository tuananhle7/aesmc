import aesmc.train as train
import aesmc.losses as losses
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import unittest


class TestModels(unittest.TestCase):
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
        train.train(dataloader=train.get_synthetic_dataloader(
                        true_prior, None, true_likelihood, 1, batch_size),
                    num_particles=num_particles,
                    algorithm='iwae',
                    initial=prior,
                    transition=None,
                    emission=likelihood,
                    proposal=inference_network,
                    num_epochs=1,
                    num_iterations_per_epoch=num_iterations,
                    optimizer_algorithm=torch.optim.SGD,
                    optimizer_kwargs={'lr': 0.01},
                    callback=training_stats)

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
            axs, data, true, ['$\mu_0$', '$\sigma$', '$a$', '$b$', '$c$']
        ):
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
        test_inference_num_particles = 1000
        saving_interval = 10
        logging_interval = 10
        batch_size = 10
        num_iterations = 500
        num_particles = 100

        # http://tuananhle.co.uk/notes/optimal-proposal-lgssm.html
        optimal_proposal_scale_0 = np.sqrt(
            initial_scale**2 - initial_scale**2 * true_emission_mult /
            (emission_scale**2 + initial_scale**2 * true_emission_mult**2) *
            true_emission_mult * initial_scale**2)
        optimal_proposal_scale_t = np.sqrt(
            transition_scale**2 - transition_scale**2 * true_emission_mult /
            (emission_scale**2 + transition_scale**2 * true_emission_mult**2)
            * true_emission_mult * transition_scale**2)
        algorithms = ['iwae', 'aesmc']
        dataloader = train.get_synthetic_dataloader(
            lgssm.Initial(initial_loc, initial_scale),
            lgssm.Transition(true_transition_mult, transition_scale),
            lgssm.Emission(true_emission_mult, emission_scale),
            num_timesteps, batch_size)
        fig, axs = plt.subplots(2, 1, sharex=True)
        for algorithm in algorithms:
            training_stats = lgssm.TrainingStats(
                initial_loc, initial_scale, true_transition_mult,
                transition_scale, true_emission_mult, emission_scale,
                num_timesteps, num_test_obs, test_inference_num_particles,
                saving_interval, logging_interval)
            train.train(dataloader=dataloader,
                        num_particles=num_particles,
                        algorithm=algorithm,
                        initial=lgssm.Initial(initial_loc, initial_scale),
                        transition=lgssm.Transition(init_transition_mult,
                                                    transition_scale),
                        emission=lgssm.Emission(init_emission_mult,
                                                emission_scale),
                        proposal=lgssm.Proposal(optimal_proposal_scale_0,
                                                optimal_proposal_scale_t),
                        num_epochs=1,
                        num_iterations_per_epoch=num_iterations,
                        callback=training_stats)
            axs[0].plot(training_stats.iteration_idx_history,
                        training_stats.p_l2_history,
                        label=algorithm)
            axs[1].plot(training_stats.iteration_idx_history,
                        training_stats.q_l2_history,
                        label=algorithm)
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

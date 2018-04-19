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



        num_particles = 2
        batch_size = 10
        num_iterations = 2000

        test_cases = [(dgm.autoencoder.AutoencoderAlgorithm.IWAE, ae.DiscreteGradientEstimator.REINFORCE),\
                        (dgm.autoencoder.AutoencoderAlgorithm.IWAE, ae.DiscreteGradientEstimator.VIMCO),\
                        (dgm.autoencoder.AutoencoderAlgorithm.WAKE_SLEEP, ae.DiscreteGradientEstimator.IGNORE)]

        training_stats = [gaussian.TrainingStats(logging_interval=500) for _ in range(len(test_cases))]

        print('\nTraining the \"gaussian\" autoencoder.')
        for idx, (algo, estimator) in enumerate(test_cases):
            prior = gaussian.Prior(prior_mean_init, prior_std)
            likelihood = gaussian.Likelihood(obs_std_init)
            inference_network = gaussian.InferenceNetwork(
                q_init_mult, q_init_bias, q_init_std
            )
            autoencoder = ae.AutoEncoder(
                prior, None, likelihood, inference_network
            )
            dgm.train.train_autoencoder(
                autoencoder,
                dgm.train.get_synthetic_dataloader(
                    true_prior, None, true_likelihood, 1, batch_size
                ),
                autoencoder_algorithm=algo,
                num_epochs=1,
                num_iterations_per_epoch=num_iterations,
                num_particles=num_particles,
                wake_sleep_mode=ae.WakeSleepAlgorithm.WW,
                optimizer_algorithm=torch.optim.SGD,
                optimizer_kwargs={'lr': 0.01},
                discrete_gradient_estimator=estimator,
                callback=training_stats[idx]
            )

        fig, axs = plt.subplots(5, len(test_cases), sharex=True, sharey=True)
        fig.set_size_inches(10, 8)

        all_mean = [training_stats[i].prior_mean_history for i in range(len(test_cases))]
        all_obs = [training_stats[i].obs_std_history for i in range(len(test_cases))]
        all_mult = [training_stats[i].q_mult_history for i in range(len(test_cases))]
        all_bias = [training_stats[i].q_bias_history for i in range(len(test_cases))]
        all_std = [training_stats[i].q_std_history for i in range(len(test_cases))]
        all_data =  all_mean + all_obs + all_mult + all_bias + all_std

        for ax, data, true, ylabel in zip(
            axs.flatten(),
            all_data,
            [true_prior_mean for _ in range(len(test_cases))] + \
            [true_obs_std for _ in range(len(test_cases))] + \
            [q_true_mult for _ in range(len(test_cases))] + \
            [q_true_bias for _ in range(len(test_cases))] + \
            [q_true_std for _ in range(len(test_cases))],
            ['$\mu_0$', '', '', '$\sigma$', '', '', '$a$', '', '', '$b$', '', '', '$c$', '','']
        ):
            ax.plot(training_stats[0].iteration_idx_history, data)
            ax.axhline(true, color='black')
            ax.set_ylabel(ylabel)
            #  self.assertAlmostEqual(data[-1], true, delta=1e-1)

        axs[-1][0].set_xlabel('Iteration')
        axs[-1][1].set_xlabel('Iteration')
        axs[-1][2].set_xlabel('Iteration')
        axs[0][0].set_title('Reinforce')
        axs[0][1].set_title('VIMCO')
        axs[0][2].set_title('Wake Wake')
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

        num_particles = 5
        batch_size = 100
        num_iterations = 20000

        test_cases = [(dgm.autoencoder.AutoencoderAlgorithm.IWAE, ae.DiscreteGradientEstimator.REINFORCE),\
                        (dgm.autoencoder.AutoencoderAlgorithm.IWAE, ae.DiscreteGradientEstimator.VIMCO),\
                        (dgm.autoencoder.AutoencoderAlgorithm.WAKE_SLEEP, ae.DiscreteGradientEstimator.IGNORE)]

        training_stats = [gmm.TrainingStats(logging_interval=1000) for _ in range(len(test_cases))]


        for idx, (algo, estimator) in enumerate(test_cases):
            print('\nTraining the \"gmm\" autoencoder.')
            likelihood = gmm.Likelihood(mean_multiplier, stds)
            prior = gmm.Prior(
                init_mixture_probs_pre_softmax=init_mixture_probs_pre_softmax,
                softmax_multiplier=softmax_multiplier
            )
            inference_network = gmm.InferenceNetwork(num_mixtures)

            autoencoder = dgm.autoencoder.AutoEncoder(
                prior, None, likelihood, inference_network
            )
            dataloader = dgm.train.get_synthetic_dataloader(
                true_prior, None, likelihood, 1, batch_size
            )
            dgm.train.train_autoencoder(
                autoencoder,
                dataloader,
                autoencoder_algorithm=algo,
                num_epochs=1,
                num_iterations_per_epoch=num_iterations,
                num_particles=num_particles,
                wake_sleep_mode=ae.WakeSleepAlgorithm.WW,
                discrete_gradient_estimator=estimator,
                callback=training_stats[idx]
            )

        num_test_data = 100
        gmm_stats = [ gmm.get_stats(
            training_stats[i].mixture_probs_history,
            training_stats[i].inference_network_state_dict_history,
            true_prior, likelihood, num_test_data
        ) for i in range(len(test_cases))]
        priors = [ a for (a, _) in gmm_stats]
        posteriors = [ b for (_, b) in gmm_stats]

        # Plotting
        fig, axs = plt.subplots(2, len(test_cases), sharex=True, sharey=True)
        fig.set_size_inches(12, 4)

        for idx, (ax, data, ylabel) in enumerate(zip(
            axs.flatten(),
            priors + posteriors,
            ['$|| p_{\\theta}(z) - p_{\\theta^*}(z) ||$', '', '',
             'Avg. test\n$|| q_\phi(z | x) - p_{\\theta^*}(z | x) ||$', '', '']
        )):
            ax.plot(training_stats[0].iteration_idx_history, data)
            ax.set_ylabel(ylabel)
            ax.set_yscale('log')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        axs[-1][0].set_xlabel('Iteration')
        axs[-1][1].set_xlabel('Iteration')
        axs[-1][2].set_xlabel('Iteration')
        axs[0][0].set_title('Reinforce')
        axs[0][1].set_title('VIMCO')
        axs[0][2].set_title('Wake Wake')
        fig.tight_layout()
        filename = './test/test_autoencoder_plots/gmm.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('\nPlot saved to {}'.format(filename))

        self.assertAlmostEqual(priors[2][-1], 0, delta=1e-1)
        self.assertAlmostEqual(posteriors[2][-1], 0, delta=6e-1)

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
        num_iterations = 500
        num_particles = 10

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

    def test_vimco(self):
        from .models import gmm
        num_mixtures = 2
        mean_multiplier = 10
        stds = np.array([5 for _ in range(num_mixtures)])
        softmax_multiplier = 0.5
        num_particles = 5
        batch_size = 20000
        num_mc_samples = 100

        temp = np.arange(num_mixtures) + 5
        true_mixture_probs = temp / np.sum(temp)
        true_prior = gmm.Prior(
            init_mixture_probs_pre_softmax=np.log(
                true_mixture_probs
            ) / softmax_multiplier,
            softmax_multiplier=softmax_multiplier
        )
        likelihood = gmm.Likelihood(mean_multiplier, stds)
        inference_network = gmm.InferenceNetwork(num_mixtures, hidden_dim=1)
        autoencoder = dgm.autoencoder.AutoEncoder(
            true_prior, None, likelihood, inference_network
        )
        dataloader = dgm.train.get_synthetic_dataloader(
            true_prior, None, likelihood, 1, batch_size
        )
        discrete_gradient_estimators = [
            dgm.autoencoder.DiscreteGradientEstimator.REINFORCE,
            dgm.autoencoder.DiscreteGradientEstimator.VIMCO,
            dgm.autoencoder.DiscreteGradientEstimator.IGNORE
        ]
        observations = next(iter(dataloader))
        mean_stds = {}
        for discrete_gradient_estimator in discrete_gradient_estimators:
            mean_std_accum = MeanStdAccum()
            for mc_idx in range(num_mc_samples):
                autoencoder.zero_grad()
                loss = -torch.mean(autoencoder.forward(
                    observations, num_particles, dgm.autoencoder.AutoencoderAlgorithm.IWAE,
                    discrete_gradient_estimator
                ))
                loss.backward()
                mean_std_accum.update([p.grad for p in autoencoder.proposal.parameters()])
            mean_stds[discrete_gradient_estimator] = mean_std_accum.avg_of_means_stds()

        np.testing.assert_allclose(
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.REINFORCE][0],
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.VIMCO][0],
            rtol=1e-1
        )
        self.assertTrue(
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.REINFORCE][1] >
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.VIMCO][1]
        )
        self.assertFalse(
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.IGNORE][1] ==
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.REINFORCE][1]
        )
        self.assertFalse(
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.IGNORE][1] ==
            mean_stds[dgm.autoencoder.DiscreteGradientEstimator.VIMCO][1]
        )


if __name__ == '__main__':
    unittest.main()

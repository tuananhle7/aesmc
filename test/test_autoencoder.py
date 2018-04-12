import dgm.autoencoder as ae
import dgm.model as model
import dgm.state as st
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import unittest


class MyInitialNetwork(model.InitialNetwork):
    def __init__(self, initial_mean):
        super(MyInitialNetwork, self).__init__()
        self.mean = nn.Parameter(torch.Tensor([initial_mean]))
        self.std = 1

    def initial(self):
        return torch.distributions.Normal(loc=self.mean, scale=self.std)


class MyTransitionNetwork(model.TransitionNetwork):
    def __init__(self, transition_multiplier):
        super(MyTransitionNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([transition_multiplier]))
        self.std = 1

    def transition(self, previous_latent=None, time=None):
        return torch.distributions.Normal(
            loc=self.multiplier * previous_latent,
            scale=self.std
        )


class MyEmissionNetwork(model.EmissionNetwork):
    def __init__(self, emission_multiplier):
        super(MyEmissionNetwork, self).__init__()
        self.multiplier = nn.Parameter(torch.Tensor([emission_multiplier]))
        self.std = 1

    def emission(self, latent=None, time=None):
        return torch.distributions.Normal(
            loc=self.multiplier * latent,
            scale=self.std
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


class TestLogAncestralIndicesProposal(unittest.TestCase):
    def test_dimensions(self):
        batch_size = 3
        num_particles = 4
        self.assertEqual(
            ae.log_ancestral_indices_proposal(
                [torch.ones(batch_size, num_particles).long()],
                [torch.rand(batch_size, num_particles),
                 torch.rand(batch_size, num_particles)]
            ).size(),
            torch.Size([batch_size])
        )
        self.assertEqual(
            ae.log_ancestral_indices_proposal(
                [], [torch.rand(batch_size, num_particles)]
            ).size(),
            torch.Size([batch_size])
        )

    def test_type(self):
        batch_size = 1
        num_particles = 1
        self.assertIsInstance(
            ae.log_ancestral_indices_proposal(
                [torch.zeros(batch_size, num_particles).long()],
                [torch.rand(batch_size, num_particles),
                 torch.rand(batch_size, num_particles)]
            ),
            torch.Tensor
        )
        self.assertIsInstance(
            ae.log_ancestral_indices_proposal(
                [], [torch.rand(batch_size, num_particles)]
            ),
            torch.Tensor
        )

    def test_value(self):
        weights = [[0.2, 0.8], [0.6, 0.4], [0.1, 0.9]]
        ancestral_indices = [[1, 0], [0, 0]]
        self.assertAlmostEqual(
            ae.log_ancestral_indices_proposal(
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


class TestLogProposal(unittest.TestCase):
    def test_dimensions(self):
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

            self.assertEqual(
                ae.log_proposal(
                    my_proposal_network,
                    None,
                    original_latents,
                    ancestral_indices,
                    log_weights
                ).size(),
                torch.Size([batch_size])
            )

    def test_value(self):
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
            ae.log_proposal(
                my_proposal_network,
                None,
                original_latents,
                ancestral_indices,
                log_weights
            ).item(),
            np.log(0.2) + np.log(0.8) + np.log(0.6) + np.log(0.6) +
            scipy.stats.norm.logpdf(1, 0, 1) +
            scipy.stats.norm.logpdf(2, 0, 1) +
            scipy.stats.norm.logpdf(3, 2, 1) +
            scipy.stats.norm.logpdf(4, 1, 1) +
            scipy.stats.norm.logpdf(5, 3, 1) +
            scipy.stats.norm.logpdf(6, 3, 1),
            places=5
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

        for gradients in ['ignore', 'reinforce', 'full_reinforce']:
            elbo = my_auto_encoder.forward(
                observations=observations,
                resample=True,
                num_particles=num_particles,
                gradients=gradients
            )
            self.assertEqual(elbo.size(), torch.Size([batch_size]))
            torch.mean(elbo).backward()
            self.assertEqual(
                my_initial_network.mean.size(),
                my_initial_network.mean.grad.size()
            )

    def test_grad(self):
        batch_size = int(1e4)
        num_particles = int(1e1)

        initial_mean = 0
        transition_multiplier = 1.2
        emission_multiplier = 0.9
        proposal_multiplier = 1.1
        observations = list(
            torch.Tensor([1, 2, 3, 4]).unsqueeze(0).expand(batch_size, -1)
        )
        num_timesteps = len(observations)

        my_initial_network = MyInitialNetwork(initial_mean)
        my_transition_network = MyTransitionNetwork(transition_multiplier)
        my_emission_network = MyEmissionNetwork(emission_multiplier)
        my_proposal_network = MyProposalNetwork(proposal_multiplier)
        my_auto_encoder = ae.AutoEncoder(
            initial=my_initial_network,
            transition=my_transition_network,
            emission=my_emission_network,
            proposal=my_proposal_network,
        )

        initial_mean_grad = {}
        transition_multiplier_grad = {}
        emission_multiplier_grad = {}
        proposal_multiplier_grad = {}
        gradients_list = ['ignore', 'full_reinforce', 'reinforce']
        for gradients in gradients_list:
            my_auto_encoder.zero_grad()
            elbo = my_auto_encoder.forward(
                observations=observations,
                resample=True,
                num_particles=num_particles,
                gradients=gradients
            )
            torch.mean(elbo).backward()
            initial_mean_grad[gradients] = \
                my_initial_network.mean.grad.item()
            transition_multiplier_grad[gradients] = \
                my_transition_network.multiplier.grad.item()
            emission_multiplier_grad[gradients] = \
                my_emission_network.multiplier.grad.item()
            proposal_multiplier_grad[gradients] = \
                my_proposal_network.multiplier.grad.item()

        # Plotting
        fig, axs = plt.subplots(nrows=1, ncols=4)
        fig.set_size_inches(16, 6)
        fig.suptitle(
            'Gradients of $\int Q(X, A) \hat z(X, A) dX dA$ with '
            '{0} particles and {1} Monte Carlo samples'.format(
                num_particles, batch_size
            )
        )

        for i, (_grad, _gradname) in enumerate(zip([
            initial_mean_grad,
            transition_multiplier_grad,
            emission_multiplier_grad,
            proposal_multiplier_grad
        ], [
            'initial_mean_grad',
            'transition_multiplier_grad',
            'emission_multiplier_grad',
            'proposal_multiplier_grad'
        ])):
            axs[i].bar(
                range(len(gradients_list)),
                list(map(
                    lambda gradients: _grad[gradients],
                    gradients_list
                )),
                tick_label=gradients_list
            )
            axs[i].set_title(_gradname)

        filename = './test/test_autoencoder_plots/test_grad.pdf'
        fig.savefig(filename, bbox_inches='tight')
        print('Plot saved to {}'.format(filename))

        for gradients_1 in gradients_list:
            for gradients_2 in gradients_list:
                if gradients_1 == gradients_2:
                    break
                for _grad, _gradname in zip([
                    initial_mean_grad,
                    transition_multiplier_grad,
                    emission_multiplier_grad,
                    proposal_multiplier_grad
                ], [
                    'initial_mean_grad',
                    'transition_multiplier_grad',
                    'emission_multiplier_grad',
                    'proposal_multiplier_grad'
                ]):
                    self.assertAlmostEqual(
                        _grad[gradients_1],
                        _grad[gradients_2],
                        msg='{0}[{1}] is not almost equal to {0}[{2}]'.format(
                            _gradname, gradients_1, gradients_2
                        )
                    )


if __name__ == '__main__':
    unittest.main()

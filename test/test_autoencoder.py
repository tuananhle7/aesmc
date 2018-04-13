import dgm.autoencoder as ae
import dgm.model as model
import dgm.state as st
import matplotlib.pyplot as plt
import numpy as np
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


if __name__ == '__main__':
    unittest.main()

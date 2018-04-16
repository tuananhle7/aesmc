import copy
import dgm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Prior(dgm.model.InitialNetwork):
    def __init__(self, init_mixture_probs_pre_softmax, softmax_multiplier=0.5):
        super(Prior, self).__init__()
        self.num_mixtures = len(init_mixture_probs_pre_softmax)
        self.mixture_probs_pre_softmax = nn.Parameter(
            torch.Tensor(init_mixture_probs_pre_softmax)
        )
        self.softmax_multiplier = softmax_multiplier

    def probs(self):
        return F.softmax(
            self.mixture_probs_pre_softmax * self.softmax_multiplier, dim=0
        )

    def initial(self):
        return torch.distributions.Categorical(probs=self.probs())


class Likelihood(dgm.model.EmissionDistribution):
    def __init__(self, mean_multiplier, stds):
        self.num_mixtures = len(stds)
        self.means = mean_multiplier * torch.arange(self.num_mixtures)
        self.stds = torch.Tensor(stds)

    def emission(self, latent=None, time=None):
        return torch.distributions.Normal(
            loc=self.means[latent],
            scale=self.stds[latent]
        )


class InferenceNetwork(dgm.model.ProposalNetwork):
    def __init__(self, num_mixtures):
        super(InferenceNetwork, self).__init__()
        self.num_mixtures = num_mixtures
        self.mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, self.num_mixtures),
            nn.Softmax(dim=1)
        )

    def probs(self, observation):
        return self.mlp(observation.unsqueeze(-1))

    def proposal(self, previous_latent=None, time=None, observations=None):
        return torch.distributions.Categorical(
            probs=self.probs(observations[0])
        )


class TrainingStats(object):
    def __init__(self, logging_interval=100):
        self.mixture_probs_history = []
        self.inference_network_state_dict_history = []
        self.iteration_idx_history = []
        self.logging_interval = logging_interval

    def __call__(self, epoch_idx, epoch_iteration_idx, autoencoder):
        self.mixture_probs_history.append(
            autoencoder.initial.probs().detach().numpy()
        )
        self.inference_network_state_dict_history.append(copy.deepcopy(
            autoencoder.proposal.state_dict()
        ))
        self.iteration_idx_history.append(epoch_iteration_idx)
        if epoch_iteration_idx % self.logging_interval == 0:
            print('Iteration: {}'.format(epoch_iteration_idx))


def get_log_evidence(prior, likelihood, observation):
    num_samples = len(observation)
    num_mixtures = prior.num_mixtures
    return dgm.math.logsumexp(
        torch.log(
            prior.probs().unsqueeze(0).expand(num_samples, -1)
        ) + torch.distributions.Normal(
            loc=likelihood.means.unsqueeze(0).expand(num_samples, -1),
            scale=likelihood.stds.unsqueeze(0).expand(num_samples, -1)
        ).log_prob(
            observation.unsqueeze(-1).expand(-1, num_mixtures)
        ),
        dim=1
    )


def get_posterior(prior, likelihood, observation):
    num_samples = len(observation)
    z = torch.arange(prior.num_mixtures).long()
    log_evidence = get_log_evidence(prior, likelihood, observation)

    z_expanded = z.unsqueeze(0).expand(num_samples, prior.num_mixtures)
    observation_expanded = observation.unsqueeze(-1).expand(
        num_samples, prior.num_mixtures
    )
    log_evidence_expanded = log_evidence.unsqueeze(-1).expand(
        num_samples, prior.num_mixtures
    )
    log_joint_expanded = prior.initial().log_prob(z_expanded) + \
        likelihood.emission(latent=z_expanded).log_prob(observation_expanded)

    return torch.exp(log_joint_expanded - log_evidence_expanded)


def get_stats(
    mixture_probs_history, inference_network_state_dict_history, true_prior,
    likelihood, num_test_data
):
    num_mixtures = true_prior.num_mixtures

    prior_l2 = []
    posterior_l2 = []

    dataloader = dgm.train.get_synthetic_dataloader(
        true_prior, None, likelihood, 1, num_test_data
    )
    test_observations = next(iter(dataloader))
    true_posterior = get_posterior(
        true_prior, likelihood, test_observations[0]
    ).detach().numpy()

    for inference_network_state_dict, mixture_probs in zip(
        inference_network_state_dict_history, mixture_probs_history
    ):
        inference_network = InferenceNetwork(num_mixtures)
        inference_network.load_state_dict(inference_network_state_dict)
        prior_l2.append(np.linalg.norm(
            mixture_probs - true_prior.probs().detach().numpy()
        ))
        posterior_l2.append(np.mean(np.linalg.norm(
            inference_network.probs(test_observations[0]).detach().numpy() -
            true_posterior,
            axis=1
        )))

    return prior_l2, posterior_l2

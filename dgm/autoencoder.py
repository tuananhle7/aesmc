from . import inference
from . import math
from . import state

import numpy as np
import torch
import torch.nn as nn


def log_ancestral_indices_proposal(ancestral_indices, log_weights):
    """Returns a log of the proposal density (on counting measure) of the
    ancestral indices.

    input:
        ancestral_indices: list of `LongTensor`s
            [batch_size, num_particles] of length (len(log_weights) - 1); can
            be empty
        log_weights: list of `Tensor`s [batch_size, num_particles]

    output: `Tensor`s [batch_size] that performs the computation

        \log\left(
            \prod_{t = 1}^{num_timesteps - 1} \prod_{k = 0}^{num_particles - 1}
                Discrete(
                    a_{t - 1}^k | w_{t - 1}^1, ...,  w_{t - 1}^{num_particles}
                )
        \right),

        on each element in the batch where

            num_timesteps = len(log_weights),
            a_{t - 1}^k = ancestral_indices[t - 1][b][k]
            w_{t - 1}^k = log_weights[t - 1][b][k]
            Discrete(a | p_1, ..., p_K) = p_a / (\sum_{k = 1}^K p_k)

        Note: returns a zero torch.Tensor [batch_size] if num_timesteps == 1.
    """

    assert(len(ancestral_indices) == len(log_weights) - 1)
    if len(ancestral_indices) == 0:
        return torch.zeros(log_weights[0].size(0))

    log_normalized_weights = math.lognormexp(
        torch.stack(log_weights, dim=0), dim=2
    )

    return torch.sum(torch.sum(torch.gather(
        log_normalized_weights[:-1],
        dim=2,
        index=torch.stack(ancestral_indices, dim=0)
    ), dim=2), dim=0)


def log_proposal(
    proposal,
    observations,
    original_latents,
    ancestral_indices,
    log_weights
):
    """Returns a log of the proposal density of both the particle values and the
    ancestral indices.

    input:
        proposal: dgm.model.ProposalDistribution object
        observations: list of `torch.Tensor`s [batch_size, dim1, ..., dimN] or
            `dict`s thereof
        original_latents: list of `torch.Tensor`s (or `dict` thereof)
            [batch_size, num_particles] of length len(observations)
        log_weights: list of `torch.Tensor`s [batch_size, num_particles]
            of length len(observations)
        ancestral_indices: list of `torch.LongTensor`s
            [batch_size, num_particles] of length (len(observations) - 1)

    output: `torch.Tensor` [batch_size] that performs the computation

        \log\left(
            {
                \prod_{k = 0}^{num_particles} q_0(x_0^k)
            }

            *

            {
                \prod_{t = 1}^{num_timesteps - 1}
                \prod_{k = 0}^{num_particles - 1}
                    q_t(x_t^k | x_{t - 1}^{a_{t - 1}^k})
                    Discrete(a_{t - 1}^k | w_{t - 1}^{1:num_particles})
            }
        \right),

        on each element in the batch where
            num_timesteps = len(original_latents),
            x_t^k is the (t, k)th particle value
            a_{t - 1}^k = ancestral_indices[t - 1][b][k]
            w_{t - 1}^k = log_weights[t - 1][b][k]
            q_0(x_0) is the initial proposal density
            q_t(x_t | x_{t - 1}) is the intermediate proposal density
            Discrete(a | p_1, ..., p_K) = p_a / (\sum_{k = 1}^K p_k)
    """
    assert(len(log_weights) == len(original_latents))
    assert(len(ancestral_indices) == len(original_latents) - 1)

    log_particle_value_proposal = 0
    previous_latent = None
    for time in range(len(original_latents)):
        latent = original_latents[time]

        # Invariant at this point:
        # latent contains x_t^k
        # previous_latent contains x_{t - 1}^{a_{t - 1}^k}
        #   (or None if time==0)

        _proposal = proposal.proposal(
            previous_latent=previous_latent,
            time=time,
            observations=observations
        )
        log_particle_value_proposal += torch.sum(
            state.log_prob(_proposal, latent), dim=1
        )

        # Invariant at this point:
        # log_particle_value_proposal contains
        # \sum_{k = 0}^{num_particles - 1} \log q_0(x_0^{a_0^k}) +
        # \sum_{t = 1}^{time} \sum_{k = 0}^{num_particles - 1}
        #   \log q_t(x_t^k | x_{t - 1}^{a_{t - 1}^k})

        if time < len(original_latents) - 1:
            previous_latent = state.resample(
                original_latents[time],
                ancestral_indices[time]
            )

    return log_ancestral_indices_proposal(ancestral_indices, log_weights) + \
        log_particle_value_proposal


class AutoEncoder(nn.Module):
    def __init__(self, initial, transition, emission, proposal):
        super(AutoEncoder, self).__init__()
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.proposal = proposal

    def forward_full_reinforce(self, observations, num_particles):
        """Evaluate a computation graph whose gradient is an estimator for the
        gradient of the ELBO using the Reinforce trick for both particle values
        and ancestral indices.
        """

        inference_result = inference.infer(
            algorithm='smc',
            observations=observations,
            initial=self.initial,
            transition=self.transition,
            emission=self.emission,
            proposal=self.proposal,
            num_particles=num_particles,
            reparameterized=False,
            return_log_marginal_likelihood=True,
            return_latents=False,
            return_original_latents=True,
            return_log_weight=False,
            return_log_weights=True,
            return_ancestral_indices=True
        )

        log_proposal_ = log_proposal(
            self.proposal,
            observations,
            list(map(
                lambda original_latent: original_latent.clone(),
                inference_result['original_latents']
            )),
            list(map(
                lambda ancestral_index: ancestral_index.clone(),
                inference_result['ancestral_indices']
            )),
            list(map(
                lambda log_weight: log_weight.clone(),
                inference_result['log_weights']
            ))
        )

        return inference_result['log_marginal_likelihood'] + \
            log_proposal_ * \
            inference_result['log_marginal_likelihood'].detach()

    def forward_ignore(self, observations, num_particles):
        """Evaluate a computation graph that returns the log marginal likelihood
        estimator which has been sampled using reparameterized raw samples for
        particle values and non-reparameterized, non-Reinforced samples for
        ancestral indices.
        """

        return inference.infer(
            algorithm='smc',
            observations=observations,
            initial=self.initial,
            transition=self.transition,
            emission=self.emission,
            proposal=self.proposal,
            num_particles=num_particles,
            reparameterized=True,
            return_log_marginal_likelihood=True,
            return_latents=False,
            return_original_latents=False,
            return_log_weight=False,
            return_log_weights=False,
            return_ancestral_indices=False
        )['log_marginal_likelihood']

    def forward_reinforce(self, observations, num_particles):
        """Evaluate a computation graph whose gradient is an estimator for the
        gradient of the ELBO using the reparameterization trick for particle
        values and the Reinforce trick for the ancestral indices.
        """

        inference_result = inference.infer(
            algorithm='smc',
            observations=observations,
            initial=self.initial,
            transition=self.transition,
            emission=self.emission,
            proposal=self.proposal,
            num_particles=num_particles,
            reparameterized=True,
            return_log_marginal_likelihood=True,
            return_latents=False,
            return_original_latents=False,
            return_log_weight=False,
            return_log_weights=True,
            return_ancestral_indices=True
        )
        log_ancestral_indices_proposal_ = log_ancestral_indices_proposal(
            inference_result['ancestral_indices'],
            inference_result['log_weights']
        )

        return log_ancestral_indices_proposal_ * \
            inference_result['log_marginal_likelihood'].detach() + \
            inference_result['log_marginal_likelihood']

    def forward(
        self, observations, resample, num_particles, gradients='ignore'
    ):
        """Evaluate a computation graph whose gradient is an estimator for the
        gradient of the ELBO.

        input:
            observations: list of `torch.Tensor`s [batch_size, dim1, ..., dimN]
                or `dict`s thereof
            resample: bool
            gradients: only applicable when resample is True; either 'ignore'
                or 'reinforce' or 'full_reinforce' (default: 'ignore')

        output: torch.Tensor [batch_size]
        """

        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        if resample:
            if gradients == 'full_reinforce':
                return self.forward_full_reinforce(observations, num_particles)
            elif gradients == 'ignore':
                return self.forward_ignore(observations, num_particles)
            elif gradients == 'reinforce':
                return self.forward_reinforce(observations, num_particles)
            else:
                raise ValueError(
                    "gradients argument must be either 'ignore', 'reinforce'"
                    " or 'full_reinforce'; received: {}".format(gradients)
                )
        else:
            return inference.infer(
                algorithm='is',
                observations=observations,
                initial=self.initial,
                transition=self.transition,
                emission=self.emission,
                proposal=self.proposal,
                num_particles=num_particles,
                reparameterized=True,
                return_log_marginal_likelihood=True,
                return_latents=False,
                return_original_latents=False,
                return_log_weight=False,
                return_log_weights=False,
                return_ancestral_indices=False
            )['log_marginal_likelihood']

from . import math
from . import state
from .util import *

import enum
import numpy as np
import torch
import pdb


class WakeSleepAlgorithm(enum.Enum):
    IGNORE = 0
    WS = 1  # wake theta sleep phi
    WW = 2  # wake theta wake phi
    WSW = 3 
    WSWA = 4

def logsumexp(values, dim=0, keepdim=False):
    """Logsumexp of a Tensor/Variable.
    See https://en.wikipedia.org/wiki/LogSumExp.
    input:
        values: Tensor/Variable [dim_1, ..., dim_N]
        dim: n
    output: result Tensor/Variable
        [dim_1, ..., dim_{n - 1}, dim_{n + 1}, ..., dim_N] where
        result[i_1, ..., i_{n - 1}, i_{n + 1}, ..., i_N] =
            log(sum_{i_n = 1}^N exp(values[i_1, ..., i_N]))
    """

    values_max, _ = torch.max(values, dim=dim, keepdim=True)
    result = values_max + torch.log(torch.sum(
        torch.exp(values - values_max), dim=dim, keepdim=True
    ))
    return result if keepdim else result.squeeze(dim)

def infer(
    wake_sleep_mode,
    observations,
    initial,
    transition,
    emission,
    proposal,
    num_particles,
    return_log_marginal_likelihood=False,
    return_latents=True,
    return_log_weight=True,
    return_log_weights=False
):

    batch_size = next(iter(observations[0].values())).size(0) \
        if isinstance(observations[0], dict) else observations[0].size(0)

    log_weights = []
    latents = []

    _proposal = proposal.proposal(time=0, observations=observations)
    latent = state.sample(_proposal, batch_size, num_particles)
    proposal_log_prob = state.log_prob(_proposal, latent)
    initial_log_prob = state.log_prob(initial.initial(), latent)
    emission_log_prob = state.log_prob(
        emission.emission(latent=latent, time=0),
        expand_observation(observations[0], num_particles)
    )

    log_weights.append(
        initial_log_prob + emission_log_prob - proposal_log_prob
    )
    if return_latents:
        latents.append(latent)

    for time in range(1, len(observations)):
        previous_latent = latent

        _proposal = proposal.proposal(
            previous_latent=previous_latent,
            time=time,
            observations=observations
        )
        latent = state.sample(_proposal, batch_size, num_particles)
        proposal_log_prob = state.log_prob(_proposal, latent)
        transition_log_prob = state.log_prob(
            transition.transition(previous_latent=previous_latent, time=time),
            latent
        )
        emission_log_prob = state.log_prob(
            emission.emission(latent=latent, time=time),
            expand_observation(observations[time], num_particles)
        )

        if return_latents:
            latents.append(latent)

        log_weights.append(
            transition_log_prob + emission_log_prob - proposal_log_prob
        )

    if return_log_marginal_likelihood:
        log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
        log_marginal_likelihood = math.logsumexp(log_weight, dim=1) - \
            np.log(num_particles)
    else:
        log_marginal_likelihood = None

    if return_log_weight:
        if not return_log_marginal_likelihood:
            # already calculated above
            log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
    else:
        log_weight = None

    if not return_log_weights:
        log_weights = None

    if (wake_sleep_mode == WakeSleepAlgorithm.IGNORE):
        return {
            'log_marginal_likelihood': log_marginal_likelihood,
            'latents': latents,
            'log_weight': log_weight,
            'log_weights': log_weights,
        }

    elif (wake_sleep_mode == WakeSleepAlgorithm.WS):
        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        log_probs = []
        latents = []

        _initial = state.sample(initial.initial(), batch_size, num_particles)
        _emission = state.sample(emission.emission(latent=_initial, time=0), batch_size, num_particles)

        proposal_log_prob = state.log_prob(proposal.proposal(time=0, observations=_emission), _initial)
        previous_latent = _initial.detach()

        log_probs.append(proposal_log_prob)
        observations = _emission.unsqueeze(0).detach()
        for time in range(1, len(observations)):
            _next_latent = state.sample(transition.transition(previous_latent=previous_latent, time=time), batch_size, num_particles)
            _emission = state.sample(emission.emission(latent=_next_latent, time=time), batch_size, num_particles)
            observations = torch.cat((observations, _emission.unsqueeze(0).detach()), 0)
            proposal_log_prob = state.log_prob(proposal.proposal(time=time, observations=observations), _next_latent)
            previous_latent = _next_latent.detach()
        
            log_probs.append(proposal_log_prob)
        
        if return_log_marginal_likelihood:
            log_prob = torch.sum(torch.stack(log_probs, dim=0), dim=0)
            log_marginal_likelihood = math.logsumexp(log_prob, dim=1) - \
                np.log(num_particles)

        return {
            'log_marginal_likelihood': log_marginal_likelihood,
            'latents': latents,
            'log_weight': log_prob,
            'log_weights': log_probs,
        }

    elif (wake_sleep_mode == WakeSleepAlgorithm.WW):
        return 0

    return {
        'test': [0],
    }



from .math import *
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

def infer(
    wake_sleep_mode,
    observations,
    initial,
    transition,
    emission,
    proposal,
    num_particles,
    evidence=None,
    return_log_marginal_likelihood=False,
    return_latents=True,
    return_log_weight=True,
    return_log_weights=False
):

    #  batch_size = next(iter(observations[0].values())).size(0) \
    #      if isinstance(observations[0], dict) else observations[0].size(0)
    #
    #  log_weights = []
    #  latents = []
    #
    #  _proposal = proposal.proposal(time=0, observations=observations)
    #  latent = state.sample(_proposal, batch_size, num_particles)
    #  proposal_log_prob = state.log_prob(_proposal, latent)
    #  initial_log_prob = state.log_prob(initial.initial(), latent)
    #  emission_log_prob = state.log_prob(
    #      emission.emission(latent=latent, time=0),
    #      expand_observation(observations[0], num_particles)
    #  )
    #
    #  log_weights.append(
    #      initial_log_prob + emission_log_prob - proposal_log_prob
    #  )
    #  if return_latents:
    #      latents.append(latent)
    #
    #  for time in range(1, len(observations)):
    #      previous_latent = latent
    #
    #      _proposal = proposal.proposal(
    #          previous_latent=previous_latent,
    #          time=time,
    #          observations=observations
    #      )
    #      latent = state.sample(_proposal, batch_size, num_particles)
    #      proposal_log_prob = state.log_prob(_proposal, latent)
    #      transition_log_prob = state.log_prob(
    #          transition.transition(previous_latent=previous_latent, time=time),
    #          latent
    #      )
    #      emission_log_prob = state.log_prob(
    #          emission.emission(latent=latent, time=time),
    #          expand_observation(observations[time], num_particles)
    #      )
    #
    #      if return_latents:
    #          latents.append(latent)
    #
    #      log_weights.append(
    #          transition_log_prob + emission_log_prob - proposal_log_prob
    #      )
    #
    #  if return_log_marginal_likelihood:
    #      log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
    #      log_marginal_likelihood = math.logsumexp(log_weight, dim=1) - \
    #          np.log(num_particles)
    #  else:
    #      log_marginal_likelihood = None
    #
    #  if return_log_weight:
    #      if not return_log_marginal_likelihood:
    #          # already calculated above
    #          log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
    #  else:
    #      log_weight = None
    #
    #  if not return_log_weights:
    #      log_weights = None

    if (wake_sleep_mode == WakeSleepAlgorithm.WS):
        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        log_probs = []
        latents = []

        _initial = state.sample(initial.initial(), batch_size, num_particles)
        _emission = state.sample(emission.emission(latent=_initial, time=0), batch_size, num_particles)

        proposal_log_prob = state.log_prob(proposal.proposal(time=0, observations=_emission), _initial)
        previous_latent = _initial.detach()

        log_probs.append(proposal_log_prob)
        samples = _emission.unsqueeze(0).detach()
        #  samples = [_emission.detach()]
        for time in range(1, len(observations)):
            _next_latent = state.sample(transition.transition(previous_latent=previous_latent, time=time), batch_size, num_particles)
            _emission = state.sample(emission.emission(latent=_next_latent, time=time), batch_size, num_particles)
            samples = torch.cat((samples, _emission.unsqueeze(0).detach()), 0)
            #  samples.append(_emission.detach())
            proposal_log_prob = state.log_prob(proposal.proposal(time=time, observations=samples, previous_latent=previous_latent), _next_latent)
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
        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        log_infs = []
        log_weights = []
        latents = []

        _proposal = proposal.proposal(time=0, observations=observations)
        latent = state.sample(_proposal, batch_size, num_particles)
        proposal_log_prob = state.log_prob(_proposal, latent)

        log_q = state.log_prob(_proposal, latent)

        log_infs.append(log_q)

        initial_log_prob = state.log_prob(initial.initial(), latent)
        emission_log_prob = state.log_prob(
            emission.emission(latent=latent, time=0),
            expand_observation(observations[0], num_particles)
        )

        evidence_log_prob = 0 if evidence is None \
                            else state.log_prob(
                                    evidence.emission(time=0), 
                                    expand_observation(observations[0], num_particles))

        log_weights.append(
            initial_log_prob + emission_log_prob - proposal_log_prob - evidence_log_prob
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
            log_q = state.log_prob(_proposal, latent)

            log_infs.append(log_q)

            transition_log_prob = state.log_prob(
                transition.transition(previous_latent=previous_latent, time=time),
                latent
            )
            emission_log_prob = state.log_prob(
                emission.emission(latent=latent, time=time),
                expand_observation(observations[time], num_particles)
            )

            evidence_log_prob = 0 if evidence is None \
                                else state.log_prob(
                                        evidence.emission(time=time), 
                                        expand_observation(observations[time], num_particles))

            if return_latents:
                latents.append(latent)

            log_weights.append(
                transition_log_prob + emission_log_prob - proposal_log_prob - evidence_log_prob
            )

        if return_log_marginal_likelihood:
            log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
            normalized_log_weight = torch.exp(lognormexp(log_weight, dim=1))
            log_q = torch.sum(torch.stack(log_infs, dim=0), dim=0)
            log_marginal_likelihood = normalized_log_weight.detach() * log_q
        else:
            log_marginal_likelihood = None

        return {
            'log_marginal_likelihood': log_marginal_likelihood,
            'latents': latents,
            'log_weight': log_weight,
            'log_weights': log_weights,
        }

    raise TypeError('inference_algorithm must be an InferenceAlgorithm')

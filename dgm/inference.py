from . import math
from . import state
from .util import *
from . import autoencoder as ae

import enum
import numpy as np
import torch


class InferenceAlgorithm(enum.Enum):
    IS = 0  # importance sampling
    SMC = 1  # sequential Monte Carlo
    WS = 2 # variant of ws


def mixture_sample_and_log_prob(proposal, sample, mixture_probs, sample_range=2):
    sample = torch.Tensor.float(sample)
    num_samples = len(sample.contiguous().view(-1))
    uniform_samples = torch.Tensor.float(torch.multinomial(torch.ones(sample_range), num_samples, replacement=True))
    indices = torch.multinomial(mixture_probs, num_samples, replacement=True)

    samples = torch.cat([sample.contiguous().view(-1).unsqueeze(1), uniform_samples.unsqueeze(1)], dim=1)
    mixture_samples = torch.gather(samples, dim=1, index=indices.unsqueeze(-1)).squeeze(-1).view(sample.size())

    log_sample = state.log_prob(proposal, sample)
    log_pdfs = torch.cat([log_sample.contiguous().view(-1).unsqueeze(1), -torch.log(torch.Tensor([sample_range]).expand(num_samples)).unsqueeze(1)], dim=1) 
    log_mixture_pdfs = math.logsumexp(log_pdfs + torch.log(mixture_probs), dim=1).view(sample.size())

    return mixture_samples, log_mixture_pdfs


def get_resampled_latents(latents, ancestral_indices):
    """Resample list of latents.

    input:
        latents: list of `torch.Tensor`s [batch_size, num_particles] or `dict`
            thereof
        ancestral_indices: list where each element is a LongTensor
            [batch_size, num_particles] of length (len(latents) - 1); can
            be empty.

    output: list of elements of the same type as latents
    """

    assert(len(ancestral_indices) == len(latents) - 1)
    if isinstance(latents[0], dict):
        temp_value = next(iter(latents[0].values()))
    else:
        temp_value = latents[0]
    batch_size, num_particles = temp_value.size()[:2]

    if temp_value.is_cuda:
        resampled_ancestral_index = torch.arange(0, num_particles).long().\
            cuda().unsqueeze(0).expand(batch_size, num_particles)
    else:
        resampled_ancestral_index = torch.arange(0, num_particles).long().\
            unsqueeze(0).expand(batch_size, num_particles)

    result = []
    for idx, latent in reversed(list(enumerate(latents))):
        result.insert(0, state.resample(latent, resampled_ancestral_index))
        if idx != 0:
            resampled_ancestral_index = torch.gather(
                ancestral_indices[idx - 1],
                dim=1,
                index=resampled_ancestral_index
            )

    return result


def sample_ancestral_index(log_weight):
    """Sample ancestral index using systematic resampling.

    input:
        log_weight: log of unnormalized weights, `torch.Tensor`
            [batch_size, num_particles]
    output:
        zero-indexed ancestral index: LongTensor [batch_size, num_particles]
    """

    if torch.sum(log_weight != log_weight).item() != 0:
        raise FloatingPointError('log_weight contains nan element(s)')

    batch_size, num_particles = log_weight.size()
    indices = np.zeros([batch_size, num_particles])

    uniforms = np.random.uniform(size=[batch_size, 1])
    pos = (uniforms + np.arange(0, num_particles)) / num_particles

    normalized_weights = math.exponentiate_and_normalize(
        log_weight.detach().cpu().numpy(),
        dim=1
    )

    # np.ndarray [batch_size, num_particles]
    cumulative_weights = np.cumsum(normalized_weights, axis=1)

    # hack to prevent numerical issues
    cumulative_weights = cumulative_weights / np.max(
        cumulative_weights,
        axis=1,
        keepdims=True
    )

    for batch in range(batch_size):
        indices[batch] = np.digitize(pos[batch], cumulative_weights[batch])

    if log_weight.is_cuda:
        return torch.from_numpy(indices).long().cuda()
    else:
        return torch.from_numpy(indices).long()

def infer(
    inference_algorithm,
    wake_sleep_mode,
    wake_optimizer,
    sleep_optimizer,
    observations,
    initial,
    transition,
    emission,
    proposal,
    num_particles,
    return_log_marginal_likelihood=False,
    return_latents=True,
    return_original_latents=False,
    return_log_weight=True,
    return_log_weights=False,
    return_ancestral_indices=False
):
    """Perform inference on a state space model using either sequential Monte
    Carlo or importance sampling.

    input:
        inference_algorithm: InferenceAlgorithm value
        observations: list of `torch.Tensor`s [batch_size, dim1, ..., dimN] or
            `dict`s thereof
        initial: dgm.model.InitialDistribution object
        transition: dgm.model.TransitionDistribution object
        emission: dgm.model.EmissionDistribution object
        proposal: dgm.model.ProposalDistribution object
        num_particles: int; number of particles
        return_log_marginal_likelihood: bool (default: False)
        return_latents: bool (default: True)
        return_original_latents: bool (default: False); only applicable
            for InferenceAlgorithm.SMC
        return_log_weight: bool (default: True)
        return_log_weights: bool (default: False)
        return_ancestral_indices: bool (default: False); only applicable for
            InferenceAlgorithm.SMC
    output:
        a dict containing key-value pairs for a subset of the following keys
        as specified by the return_{} parameters:
            log_marginal_likelihood: `torch.Tensor` [batch_size]
            latents: list of `torch.Tensor`s (or `dict` thereof)
                [batch_size, num_particles] of length len(observations)
            original_latents: list of `torch.Tensor`s (or `dict` thereof)
                [batch_size, num_particles] of length len(observations)
            log_weight: `torch.Tensor` [batch_size, num_particles]
            log_weights: list of `torch.Tensor`s [batch_size, num_particles]
                of length len(observations)
            ancestral_indices: list of `torch.LongTensor`s
                [batch_size, num_particles] of length len(observations)

        Note that (latents, log_weight) characterize the posterior.
    """
    if not isinstance(inference_algorithm, InferenceAlgorithm):
        raise TypeError('inference_algorithm must be an InferenceAlgorithm \
        enum.')

    batch_size = next(iter(observations[0].values())).size(0) \
        if isinstance(observations[0], dict) else observations[0].size(0)

    if return_original_latents or return_latents:
        original_latents = []
    if inference_algorithm == InferenceAlgorithm.SMC:
        ancestral_indices = []
    log_weights = []
    log_latents = []

    _proposal = proposal.proposal(time=0, observations=observations)
    latent = state.sample(_proposal, batch_size, num_particles)
    proposal_log_prob = state.log_prob(_proposal, latent)
    initial_log_prob = state.log_prob(initial.initial(), latent)
    emission_log_prob = state.log_prob(
        emission.emission(latent=latent, time=0),
        state.expand_observation(observations[0], num_particles)
    )

    if return_original_latents or return_latents:
        original_latents.append(latent)

    log_weights.append(
        initial_log_prob + emission_log_prob - proposal_log_prob
    )
    log_latents.append(proposal_log_prob)
    for time in range(1, len(observations)):
        if inference_algorithm == InferenceAlgorithm.SMC:
            ancestral_indices.append(sample_ancestral_index(log_weights[-1]))
            previous_latent = state.resample(latent, ancestral_indices[-1])
        else:
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
            state.expand_observation(observations[time], num_particles)
        )

        if return_original_latents or return_latents:
            original_latents.append(latent)

        log_weights.append(
            transition_log_prob + emission_log_prob - proposal_log_prob
        )
        log_latents.append(proposal_log_prob)

    if inference_algorithm == InferenceAlgorithm.SMC:
        if return_log_marginal_likelihood:
            temp = math.logsumexp(torch.stack(log_weights, dim=0), dim=2) - \
                np.log(num_particles)
            log_marginal_likelihood = torch.sum(temp, dim=0)
        else:
            log_marginal_likelihood = None

        if return_latents:
            latents = get_resampled_latents(
                original_latents, ancestral_indices
            )
        else:
            latents = None

        if not return_original_latents:
            original_latents = None

        if return_log_weight:
            log_weight = log_weights[-1]
        else:
            log_weight = None

        if not return_log_weights:
            log_weights = None

        if not return_ancestral_indices:
            ancestral_indices = None
    else:
        if return_log_marginal_likelihood:
            log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
            log_marginal_likelihood = math.logsumexp(log_weight, dim=1) - \
                np.log(num_particles)
        else:
            log_marginal_likelihood = None

        if return_latents:
            latents = original_latents
        else:
            latents = None

        original_latents = None
        if return_original_latents:
            raise RuntimeWarning('return_original_latents shouldn\'t be True\
            for InferenceAlgorithm.IS')

        if return_log_weight:
            if not return_log_marginal_likelihood:
                # already calculated above
                log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
        else:
            log_weight = None

        if not return_log_weights:
            log_weights = None

        ancestral_indices = None
        if return_ancestral_indices:
            raise RuntimeWarning('return_ancestral_indices shouldn\'t be True\
            for InferenceAlgorithm.IS')

    if inference_algorithm != InferenceAlgorithm.WS:
        return {
            'log_marginal_likelihood': log_marginal_likelihood,
            'latents': latents,
            'original_latents': original_latents,
            'log_weight': log_weight,
            'log_weights': log_weights,
            'ancestral_indices': ancestral_indices,
            'last_latent': latent
            #  'log_latents': log_latents
        }

    else:
        return sleep_loss(
                log_marginal_likelihood, 
                wake_sleep_mode, 
                wake_optimizer, 
                sleep_optimizer,
                observations, 
                initial, 
                transition, 
                emission,
                proposal,
                num_particles)

def sleep_loss(
    wake_loss,
    wake_sleep_mode,
    wake_optimizer,
    sleep_optimizer,
    observations,
    initial,
    transition,
    emission,
    proposal,
    num_particles
):
    # update theta and zero out phi
    loss = -torch.mean(wake_loss)
    loss.backward()
    wake_optimizer.step()
    sleep_optimizer.zero_grad()
    #  torch.optim.Adam(proposal.parameters()).zero_grad()

    if (wake_sleep_mode == ae.WakeSleepAlgorithm.WS):
        # TODO fix this ..
        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        log_probs = []
        latents = []

        _initial = state.sample(initial.initial(), batch_size, num_particles)
        _emission = state.sample(emission.emission(latent=_initial, time=0), batch_size, num_particles)

        proposal_log_prob = torch.Tensor(_emission.size())
        for i in range(num_particles):
            sampled_obs = _emission[:, i]
            proposal_log_prob = state.log_prob(proposal.proposal(time=0, observations=sampled_obs), _initial)

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
            'latents': None,
            'log_weight': None,
            'log_weights': None,
            'ancestral_indices': None,
            'original_latents': None
        }

    elif (wake_sleep_mode == ae.WakeSleepAlgorithm.WW):
        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        log_infs = []
        log_weights = []
        latents = []

        mixture_probs = torch.Tensor([1.0,0.0])


        _proposal = proposal.proposal(time=0, observations=observations)
        latent = state.sample(_proposal, batch_size, num_particles)
        mixture_latent = latent
        proposal_log_prob = state.log_prob(_proposal, mixture_latent)
    	#log_sample = state.log_prob(proposal, sample)
        #mixture_latent, proposal_log_prob = mixture_sample_and_log_prob(_proposal, latent, mixture_probs)

        log_q = state.log_prob(_proposal, mixture_latent)

        log_infs.append(log_q)

        initial_log_prob = state.log_prob(initial.initial(), latent)
        emission_log_prob = state.log_prob(
            emission.emission(latent=latent, time=0),
            state.expand_observation(observations[0], num_particles)
        )

        #  evidence_log_prob = 0 if evidence is None \
        #                      else state.log_prob(
        #                              evidence.emission(time=0),
        #                              expand_observation(observations[0], num_particles))
        evidence_log_prob = 0

        log_weights.append(
            initial_log_prob + emission_log_prob - proposal_log_prob - evidence_log_prob
        )

        latents.append(latent)

        for time in range(1, len(observations)):
            previous_latent = latent

            _proposal = proposal.proposal(
                previous_latent=previous_latent,
                time=time,
                observations=observations
            )
            latent = state.sample(_proposal, batch_size, num_particles)
            mixture_latent = latent
            proposal_log_prob = state.log_prob(_proposal, latent)
            #proposal_log_prob = state.log_prob(_proposal, latent)
            #mixture_latent, proposal_log_prob = mixture_sample_and_log_prob(_proposal, latent, mixture_probs)

            log_q = state.log_prob(_proposal, latent)

            log_infs.append(log_q)

            transition_log_prob = state.log_prob(
                transition.transition(previous_latent=previous_latent, time=time),
                latent
            )
            emission_log_prob = state.log_prob(
                emission.emission(latent=latent, time=time),
                state.expand_observation(observations[time], num_particles)
            )
            evidence_log_prob = 0
            
            latents.append(latent)

            log_weights.append(
                transition_log_prob + emission_log_prob - proposal_log_prob - evidence_log_prob
            )

        log_weight = torch.sum(torch.stack(log_weights, dim=0), dim=0)
        normalized_log_weight = torch.exp(math.lognormexp(log_weight, dim=1))
        log_q = torch.sum(torch.stack(log_infs, dim=0), dim=0)
        log_marginal_likelihood = torch.sum(normalized_log_weight.detach() * log_q, dim=1)

        return {
            'log_marginal_likelihood': log_marginal_likelihood,
            'latents': None,
            'log_weight': None,
            'log_weights': None,
            'ancestral_indices': None,
            'original_latents': None
            #  'log_latents': None
        }
    else: 
        raise NotImplementedError('ok')

# NOTE: This function is currently used to calculate REINFORCE estimator of
# resampling gradients.
def ancestral_indices_log_prob(ancestral_indices, log_weights):
    """Returns a log probability of the ancestral indices.

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

        Note: returns a zero `torch.Tensor` [batch_size] if num_timesteps == 1.
    """
    if ancestral_indices is None or log_weights is None:
        return 0

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


def latents_log_prob(
    proposal,
    observations,
    original_latents,
    ancestral_indices=None,
    non_reparam=False
):
    """Returns log probability of latents under the proposal.

    input:
        proposal: dgm.model.ProposalDistribution object
        observations: list of `torch.Tensor`s [batch_size, dim1, ..., dimN] or
            `dict`s thereof
        original_latents: list of `torch.Tensor`s (or `dict` thereof)
            [batch_size, num_particles] of length len(observations)
        ancestral_indices: list of `torch.LongTensor`s
            [batch_size, num_particles] of length (len(observations) - 1). If
            not supplied, it is assumed that the original_latents have been
            generated using importance sampling (and hence with no resampling).
        non_reparam: bool; if True, only returns log probability of the
            non-reparameterizable part of the latents (default: False).

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
            }
        \right)

        on each element in the batch b where
            num_timesteps = len(original_latents),
            x_t^k is the (t, k)th particle value
            a_{t - 1}^k = ancestral_indices[t - 1][b][k]
            q_0(x_0^k) is the initial proposal density
            q_t(x_t | x_{t - 1}) is an intermediate proposal density
    """
    if original_latents is None and ancestral_indices is None:
        return 0

    if ancestral_indices is None:
        result = 0
        for time in range(len(original_latents)):
            _proposal = proposal.proposal(
                previous_latent=(
                    None if (time == 0) else original_latents[time - 1]
                ),
                time=time,
                observations=observations
            )
            result += torch.sum(state.log_prob(
                _proposal, original_latents[time], non_reparam
            ), dim=1)
        return result
    else:
        result = 0
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
            result += torch.sum(
                state.log_prob(_proposal, latent, non_reparam), dim=1
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
        return result


def control_variate(
    proposal, 
    observations,
    elbo_detached, 
    num_particles,
    original_latents,
    log_weights,
    non_reparam=False
):
    result = torch.zeros(elbo_detached.size())
    for t in range(len(log_weights)):
        log_weight = log_weights[t]

        _proposal = proposal.proposal(
            previous_latent=(
                None if (t == 0) else original_latents[t - 1]
            ),
            time=t,
            observations=observations
        )
        log_proposal = state.log_prob(
                _proposal, original_latents[t], non_reparam
                )

        for i in range(num_particles):
            log_weight_ = log_weight[:, list(set(range(num_particles)).difference(set([i])))]
            control_variate = math.logsumexp(
                torch.cat([log_weight_, torch.mean(log_weight_, dim=1, keepdim=True)], dim=1),
                dim=1
                )
            result = result + (elbo_detached - control_variate.detach()) * log_proposal[:, i]
    return result

# NOTE: This function is currently unused; consider removing it.
def latents_and_ancestral_indices_log_prob(
    proposal,
    observations,
    original_latents,
    ancestral_indices,
    log_weights,
    non_reparam=False
):
    """Returns a log of the proposal density of both the particle values and the
    ancestral indices.

    input:
        proposal: dgm.model.ProposalDistribution object
        observations: list of `torch.Tensor`s [batch_size, dim1, ..., dimN] or
            `dict`s thereof
        original_latents: list of `torch.Tensor`s (or `dict` thereof)
            [batch_size, num_particles] of length len(observations)
        ancestral_indices: list of `torch.LongTensor`s
            [batch_size, num_particles] of length (len(observations) - 1)
        log_weights: list of `torch.Tensor`s [batch_size, num_particles]
            of length len(observations)
        non_reparam: bool; if True, only returns log probability of the
            non-reparameterizable part of the latents (default: False).

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
            q_0(x_0^k) is the initial proposal density
            q_t(x_t | x_{t - 1}) is the intermediate proposal density
            Discrete(a | p_1, ..., p_K) = p_a / (\sum_{k = 1}^K p_k)
    """

    return ancestral_indices_log_prob(ancestral_indices, log_weights) + \
        latents_log_prob(
            proposal,
            observations,
            original_latents,
            ancestral_indices=None,
            non_reparam=False
        )

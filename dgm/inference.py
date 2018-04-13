from . import math
from . import state

import enum
import numpy as np
import torch


class InferenceAlgorithm(enum.Enum):
    IS = 0  # importance sampling
    SMC = 1  # sequential Monte Carlo


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


def expand_observation(observation, num_particles):
    """input:
        observation: `torch.Tensor` [batch_size, dim1, ..., dimN]
        num_particles: int

    output: `torch.Tensor` [batch_size, num_particles, dim1, ..., dimN]
    """
    batch_size = observation.size(0)
    other_sizes = list(observation.size()[1:])

    return observation.unsqueeze(1).expand(
        *([batch_size, num_particles] + other_sizes)
    )


def infer(
    inference_algorithm,
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

    _proposal = proposal.proposal(time=0, observations=observations)
    latent = state.sample(_proposal, batch_size, num_particles)
    proposal_log_prob = state.log_prob(_proposal, latent)
    initial_log_prob = state.log_prob(initial.initial(), latent)
    emission_log_prob = state.log_prob(
        emission.emission(latent=latent, time=0),
        expand_observation(observations[0], num_particles)
    )

    if return_original_latents or return_latents:
        original_latents.append(latent)

    log_weights.append(
        initial_log_prob + emission_log_prob - proposal_log_prob
    )

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
            expand_observation(observations[time], num_particles)
        )

        if return_original_latents or return_latents:
            original_latents.append(latent)

        log_weights.append(
            transition_log_prob + emission_log_prob - proposal_log_prob
        )

    if inference_algorithm == InferenceAlgorithm.SMC:
        if return_log_marginal_likelihood:
            temp = math.logsumexp(
                torch.stack(log_weights, dim=0), dim=2
            ) - np.log(num_particles)
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
            log_weight = torch.sum(
                torch.stack(log_weights, dim=0),
                dim=0
            )
            log_marginal_likelihood = math.logsumexp(
                log_weight, dim=1
            ) - np.log(num_particles)
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
                log_weight = torch.sum(
                    torch.stack(log_weights, dim=0),
                    dim=0
                )
        else:
            log_weight = None

        if not return_log_weights:
            log_weights = None

        ancestral_indices = None
        if return_ancestral_indices:
            raise RuntimeWarning('return_ancestral_indices shouldn\'t be True\
            for InferenceAlgorithm.IS')

    return {
        'log_marginal_likelihood': log_marginal_likelihood,
        'latents': latents,
        'original_latents': original_latents,
        'log_weight': log_weight,
        'log_weights': log_weights,
        'ancestral_indices': ancestral_indices
    }


# NOTE: This functions is currently used to calculate REINFORCE estimator of
# resampling gradients.
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


# NOTE: This function is currently unused; consider removing it.
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

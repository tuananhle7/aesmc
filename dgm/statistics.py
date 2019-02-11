from . import inference
from . import math
from . import state
import torch


def empirical_expectation(value, log_weight, f):
    """Empirical expectation.

    input:
        value: torch.Tensor
            [batch_size, num_particles, value_dim_1, ..., value_dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]
        f: function which takes torch.Tensor
            [batch_size, value_dim_1, ..., value_dim_N] (or
            [batch_size]) and returns a torch.Tensor
            [batch_size, dim_1, ..., dim_M] (or [batch_size])
    output: empirical expectation torch.Tensor
        [batch_size, dim_1, ..., dim_M] (or [batch_size])
    """

    assert(value.size()[:2] == log_weight.size())
    normalized_weights = math.exponentiate_and_normalize(log_weight, dim=1)

    # first particle
    f_temp = f(value[:, 0])
    w_temp = normalized_weights[:, 0]
    for i in range(f_temp.dim() - 1):
        w_temp.unsqueeze_(-1)

    emp_exp = w_temp.expand_as(f_temp) * f_temp

    # next particles
    for p in range(1, normalized_weights.size(1)):
        f_temp = f(value[:, p])
        w_temp = normalized_weights[:, p]
        for i in range(f_temp.dim() - 1):
            w_temp.unsqueeze_(-1)

        emp_exp += w_temp.expand_as(f_temp) * f_temp

    return emp_exp


def empirical_mean(value, log_weight):
    """Empirical mean.

    input:
        value: torch.Tensor
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]
    output: empirical mean torch.Tensor
        [batch_size, dim_1, ..., dim_N] (or [batch_size])
    """

    return empirical_expectation(value, log_weight, lambda x: x)


def empirical_variance(value, log_weight):
    """Empirical variance.

    input:
        value: torch.Tensor
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]
    output: empirical mean torch.Tensor
        [batch_size, dim_1, ..., dim_N] (or [batch_size])
    """

    return empirical_expectation(value, log_weight, lambda x: x**2) -\
        empirical_mean(value, log_weight)**2


def log_ess(log_weight):
    """Log of Effective sample size.

    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, num_particles] (or [num_particles])

    Returns: log of effective sample size [batch_size] (or [1])
    """
    dim = 1 if log_weight.ndimension() == 2 else 0

    return 2 * math.logsumexp(log_weight, dim=dim) - \
        math.logsumexp(2 * log_weight, dim=dim)


def ess(log_weight):
    """Effective sample size.

    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, num_particles] (or [num_particles])

    Returns: effective sample size [batch_size] (or [1])
    """

    return torch.exp(log_ess(log_weight))


# NOTE: old and untested
def infer_reconstruct_predict(
    inference_algorithm, observations, initial, transition, emission, proposal,
    num_particles, num_predictions
):
    """Infer, reconstruct and predict given a generative model, observations
    and an inference algorithm.

    More efficient than calling reconstruct_observations and
    predict_observations in turn.

    Args:
        inference_algorithm: InferenceAlgorithm value
        observations: list of tensors [batch_size, dim1, ..., dimN] or
            dicts thereof
        initial: dgm.model.InitialDistribution object
        transition: dgm.model.TransitionDistribution object
        emission: dgm.model.EmissionDistribution object
        proposal: dgm.model.ProposalDistribution object
        num_particles: int; number of particles
        num_predictions: int; number of timesteps to predict
    Returns:
        latents: list of tensors [batch_size, ...] or dicts thereof
        reconstructed_observations: list of tensors
            [batch_size, dim1, ..., dimN] or dicts thereof
        predicted_latents: list of tensors [batch_size, ...] or
            dicts thereof
        predicted_observations: list of tensors
            [batch_size, dim1, ..., dimN] or dicts thereof
        log_weight: torch.Tensor [batch_size, num_particles]
    """

    batch_size = next(iter(observations[0].values())).size(0) \
        if isinstance(observations[0], dict) else observations[0].size(0)
    num_timesteps = len(observations)

    inference_result = inference.infer(
        inference_algorithm=inference_algorithm,
        observations=observations,
        initial=initial,
        transition=transition,
        emission=emission,
        proposal=proposal,
        num_particles=num_particles,
        return_log_marginal_likelihood=False,
        return_latents=True,
        return_original_latents=False,
        return_log_weight=True,
        return_log_weights=False,
        return_ancestral_indices=False)

    log_weight = inference_result['log_weight']
    latents = inference_result['latents']
    reconstructed_observations = []
    for time in range(num_timesteps):
        reconstructed_observations.append(
            state.sample(emission(latents[:time + 1], time=time),
                         batch_size, num_particles))
    predicted_observations = []
    for time in range(num_timesteps, num_timesteps + num_predictions):
        latents.append(state.sample(transition(latents, time=time),
                       batch_size, num_particles))
        predicted_observations.append(
            state.sample(emission(latents, time=time),
                         batch_size, num_particles))

    return latents[:num_timesteps], reconstructed_observations, \
        latents[num_timesteps:], predicted_observations, log_weight


# TODO: test
def sample_from_prior(initial, transition, emission, num_timesteps,
                      batch_size):
    """Samples latents and observations from prior

    Args:
        initial: dgm.model.InitialDistribution object
        transition: dgm.model.TransitionDistribution object
        emission: dgm.model.EmissionDistribution object
        num_timesteps: int
        batch_size: int

    Returns:
        latents: list of tensors (or dict thereof)
            [batch_size] of length len(observations)
        observations: list of tensors [batch_size, dim1, ..., dimN] or
            dicts thereof
    """

    latents = []
    observations = []

    for time in range(num_timesteps):
        if time == 0:
            latents.append(state.sample(initial(), batch_size, 1))
        else:
            latents.append(state.sample(transition(
                previous_latents=latents, time=time), batch_size, 1))
        observations.append(state.sample(emission(
            latents=latents, time=time), batch_size, 1))

    def squeeze_num_particles(value):
        if isinstance(value, dict):
            return {k: squeeze_num_particles(v) for k, v in value.items()}
        else:
            return value.squeeze(1)

    return tuple(map(lambda values: list(map(squeeze_num_particles, values)),
                 [latents, observations]))

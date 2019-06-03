from . import inference
from . import math
from . import state
import torch


def empirical_expectation(value, log_weight, f):
    """Empirical expectation.

    Args:
        value: torch.Tensor
            [batch_size, num_particles, value_dim_1, ..., value_dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]
        f: function which takes torch.Tensor
            [batch_size, value_dim_1, ..., value_dim_N] (or
            [batch_size]) and returns a torch.Tensor
            [batch_size, dim_1, ..., dim_M] (or [batch_size])

    Returns: empirical expectation torch.Tensor
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

    Args:
        value: torch.Tensor
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]

    Returns: empirical mean torch.Tensor
        [batch_size, dim_1, ..., dim_N] (or [batch_size])
    """

    return empirical_expectation(value, log_weight, lambda x: x)


def empirical_variance(value, log_weight):
    """Empirical variance.

    Args:
        value: torch.Tensor
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: torch.Tensor [batch_size, num_particles]
    Returns: empirical mean torch.Tensor
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

    return 2 * torch.logsumexp(log_weight, dim=dim) - \
        torch.logsumexp(2 * log_weight, dim=dim)


def ess(log_weight):
    """Effective sample size.

    Args:
        log_weight: Unnormalized log weights
            torch.Tensor [batch_size, num_particles] (or [num_particles])

    Returns: effective sample size [batch_size] (or [1])
    """

    return torch.exp(log_ess(log_weight))


# TODO: test
def sample_from_prior(initial, transition, emission, num_timesteps,
                      batch_size):
    """Samples latents and observations from prior

    Args:
        initial: a callable object (function or nn.Module) which has no
            arguments and returns a torch.distributions.Distribution or a dict
            thereof
        transition: a callable object (function or nn.Module) with signature:
            Args:
                previous_latents: list of length time where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int
            Returns: torch.distributions.Distribution or a dict thereof
        emission: a callable object (function or nn.Module) with signature:
            Args:
                latents: list of length (time + 1) where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int
            Returns: torch.distributions.Distribution or a dict thereof
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

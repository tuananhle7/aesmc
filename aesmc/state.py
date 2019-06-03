import enum
import torch
import warnings


class BatchShapeMode(enum.Enum):
    NOT_EXPANDED = 0  # the batch_shape is [...]
    BATCH_EXPANDED = 1  # the batch_shape is [batch_size, ...]
    FULLY_EXPANDED = 2  # the batch_shape is [batch_size, num_particles, ...]


def set_batch_shape_mode(distribution, batch_shape_mode):
    """Sets the BatchShapeMode property of a distribution
    explicitly."""

    distribution.batch_shape_mode = batch_shape_mode
    return distribution


def get_batch_shape_mode(distribution, batch_size=None, num_particles=None):
    """Returns the BatchShapeMode property of a distribution.
    If the property is not set explicitly, it is inferred implicitly."""

    if hasattr(distribution, 'batch_shape_mode'):
        return distribution.batch_shape_mode
    else:
        batch_shape = distribution.batch_shape

        def warn(result):
            warnings.warn(
                'Inferred batch_shape_mode ({}) of distribution ({}) might'
                ' be wrong given its batch_shape ({}), batch_size ({}) and'
                ' num_particles ({}). Consider specifying the '
                'batch_shape_mode explicitly.'.format(
                    result, distribution, batch_shape,
                    batch_size, num_particles
                ), RuntimeWarning
            )

        if len(batch_shape) == 0:
            return BatchShapeMode.NOT_EXPANDED
        elif len(batch_shape) == 1:
            if batch_shape[0] == batch_size:
                result = BatchShapeMode.BATCH_EXPANDED
                warn(result)
            else:
                result = BatchShapeMode.NOT_EXPANDED
            return result
        else:
            if batch_shape[0] == batch_size:
                if batch_shape[1] == num_particles:
                    result = BatchShapeMode.FULLY_EXPANDED
                else:
                    result = BatchShapeMode.BATCH_EXPANDED
                warn(result)
                return result
            else:
                return BatchShapeMode.NOT_EXPANDED


def sample(distribution, batch_size, num_particles):
    """Samples from a distribution given batch size and number of particles.

    Args:
        distribution: `torch.distributions.Distribution` or `dict` thereof.

            Note: the batch_shape of distribution can have one of the following
            batch shape modes: [...],
            [batch_size, ...],
            [batch_size, num_particles, ...]. The batch shape mode
            of a distribution can be set explicitly using the
            `set_batch_shape_mode` function. If not set, the batch shape mode
            is inferred, although there can be ambiguities.

        batch_size: `int`
        num_particles: `int`

    Returns: `torch.Tensor` [batch_size, num_particles, ...] or `dict` thereof
    """

    if isinstance(distribution, dict):
        return {k: sample(v, batch_size, num_particles)
                for k, v in distribution.items()}
    elif isinstance(distribution, torch.distributions.Distribution):
        batch_shape_mode = get_batch_shape_mode(distribution, batch_size,
                                                num_particles)
        if batch_shape_mode == BatchShapeMode.NOT_EXPANDED:
            sample_shape = (batch_size, num_particles,)
        elif batch_shape_mode == BatchShapeMode.BATCH_EXPANDED:
            sample_shape = (num_particles,)
        elif batch_shape_mode == BatchShapeMode.FULLY_EXPANDED:
            sample_shape = ()
        else:
            raise ValueError('batch_shape_mode {} not supported'.format(
                batch_shape_mode))

        if distribution.has_rsample:
            result = distribution.rsample(sample_shape=sample_shape)
        else:
            raise ValueError('distribution not reparameterizable')

        if batch_shape_mode == BatchShapeMode.BATCH_EXPANDED:
            return result.transpose(0, 1)
        else:
            return result
    elif isinstance(distribution, torch.Tensor):
        return distribution
    else:
        raise AttributeError(
            'distribution must be a dict or a torch.distributions.Distribution'
            '. Got: {}'.format(distribution))


def log_prob(distribution, value):
    """Log probability of value under distribution.

    Args:
        distribution: `torch.distributions.Distribution` of batch_shape either
            [batch_size, num_particles, ...] or
            [batch_size, ...] or
            [...] or `dict` thereof.
        value: `torch.Tensor` of size
            [batch_size, num_particles, ...] + distribution.event_shape
            or `dict` thereof

    Returns: `torch.Tensor` [batch_size, num_particles] or `dict` thereof
    """
    if isinstance(distribution, dict):
        return torch.sum(torch.cat([
            log_prob(v, value[k], non_reparam).unsqueeze(0)
            for k, v in distribution.items()
        ], dim=0), dim=0)
    elif isinstance(distribution, torch.distributions.Distribution):
        value_ndim = value.ndimension()
        batch_shape_ndim = len(distribution.batch_shape)
        event_shape_ndim = len(distribution.event_shape)
        value_batch_shape_ndim = value_ndim - event_shape_ndim
        if (
            (value_batch_shape_ndim == batch_shape_ndim) or
            ((value_batch_shape_ndim - 2) == batch_shape_ndim)
        ):
            distribution._validate_sample(value)
            logp = distribution.log_prob(value)
        elif (value_batch_shape_ndim - 1) == batch_shape_ndim:
            logp = distribution.log_prob(value.transpose(0, 1)).transpose(0, 1)
        else:
            raise RuntimeError(
                'Incompatible distribution.batch_shape ({}) and '
                'value.shape ({}).'.format(
                    distribution.batch_shape, value.shape))
        return torch.sum(logp.view(value.size(0), value.size(1), -1), dim=2)
    else:
        raise AttributeError(
            'distribution must be a dict or a torch.distributions.Distribution.\
            Got: {}'.format(distribution))


def resample(value, ancestral_index):
    """Resample the value without side effects.

    Args:
        value: `torch.Tensor` [batch_size, num_particles, dim_1, ..., dim_N]
            (or [batch_size, num_particles]) or `dict` thereof
        ancestral_index: `torch.LongTensor` [batch_size, num_particles]
    Returns: resampled value [batch_size, num_particles, dim_1, ..., dim_N]
        (or [batch_size, num_particles]) or `dict` thereof
    """
    if isinstance(value, dict):
        return {k: resample(v, ancestral_index)
                for k, v in value.items()}
    elif torch.is_tensor(value):
        assert(ancestral_index.size() == value.size()[:2])
        ancestral_index_unsqueezed = ancestral_index

        for _ in range(len(value.size()) - 2):
            ancestral_index_unsqueezed = \
                ancestral_index_unsqueezed.unsqueeze(-1)

        return torch.gather(value, dim=1,
                            index=ancestral_index_unsqueezed.expand_as(value))
    else:
        raise AttributeError(
            'value must be a dict or a torch.Tensor. Got: {}'.format(value))


def expand_observation(observation, num_particles):
    """Args:
        observation: `torch.Tensor` [batch_size, ...] or `dict`
            thereof
        num_particles: int

    Returns: `torch.Tensor` [batch_size, num_particles, ...] or
        `dict` thereof
    """
    if isinstance(observation, dict):
        return {k: expand_observation(v, num_particles)
                for k, v in observation.items()}
    else:
        batch_size = observation.size(0)
        other_sizes = list(observation.size()[1:])

        return observation.unsqueeze(1).expand(
            *([batch_size, num_particles] + other_sizes))

import enum
import torch
import warnings


class DistributionBatchShapeMode(enum.Enum):
    NOT_EXPANDED = 0  # the batch_shape is [dim1, ..., dimN]
    BATCH_EXPANDED = 1  # the batch_shape is [batch_size, dim1, ..., dimN]
    # the batch_shape is [batch_size, num_particles, dim1, ..., dimN]
    FULLY_EXPANDED = 2


def set_batch_shape_mode(distribution, batch_shape_mode):
    """Sets the DistributionBatchShapeMode property of a distribution
    explicitly."""

    distribution.batch_shape_mode = batch_shape_mode
    return distribution


def get_batch_shape_mode(distribution, batch_size=None, num_particles=None):
    """Returns the DistributionBatchShapeMode property of a distribution.
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
            return DistributionBatchShapeMode.NOT_EXPANDED
        elif len(batch_shape) == 1:
            if batch_shape[0] == batch_size:
                result = DistributionBatchShapeMode.BATCH_EXPANDED
                warn(result)
            else:
                result = DistributionBatchShapeMode.NOT_EXPANDED
            return result
        else:
            if batch_shape[0] == batch_size:
                if batch_shape[1] == num_particles:
                    result = DistributionBatchShapeMode.FULLY_EXPANDED
                else:
                    result = DistributionBatchShapeMode.BATCH_EXPANDED
                warn(result)
                return result
            else:
                return DistributionBatchShapeMode.NOT_EXPANDED


def sample(distribution, batch_size, num_particles):
    """Samples from a distribution given batch size and number of particles.

    input:
        distribution: `torch.distributions.Distribution` or `dict` thereof.

            Note: the batch_shape of distribution can have one of the following
            batch shape modes: [dim1, ..., dimN],
            [batch_size, dim1, ..., dimN],
            [batch_size, num_particles, dim1, ..., dimN]. The batch shape mode
            of a distribution can be set explicitly using the
            `set_batch_shape_mode` function. If not set, the batch shape mode
            is inferred, although there can be ambiguities.

        batch_size: `int`
        num_particles: `int`

    output: `torch.Tensor` [batch_size, num_particles, dim1, ..., dimN] or
        `dict` thereof
    """

    if isinstance(distribution, dict):
        return {k: sample(v, batch_size, num_particles)
                for k, v in distribution.items()}
    elif isinstance(distribution, torch.distributions.Distribution):
        batch_shape_mode = get_batch_shape_mode(
            distribution, batch_size, num_particles)
        if batch_shape_mode == DistributionBatchShapeMode.NOT_EXPANDED:
            sample_shape = (batch_size, num_particles,)
        elif batch_shape_mode == DistributionBatchShapeMode.BATCH_EXPANDED:
            sample_shape = (num_particles,)
        elif batch_shape_mode == DistributionBatchShapeMode.FULLY_EXPANDED:
            sample_shape = ()
        else:
            raise ValueError('batch_shape_mode {} not supported'.format(
                batch_shape_mode))

        if distribution.has_rsample:
            result = distribution.rsample(sample_shape=sample_shape)
        else:
            result = distribution.sample(sample_shape=sample_shape)

        if batch_shape_mode == DistributionBatchShapeMode.BATCH_EXPANDED:
            return result.transpose(0, 1)
        else:
            return result
    elif isinstance(distribution, torch.Tensor):
        return distribution
    else:
        raise AttributeError(
            'distribution must be a dict or a torch.distributions.Distribution'
            '. Got: {}'.format(distribution)
        )


# what's non_reparam for?
def log_prob(distribution, value, non_reparam=False):
    """Log probability of value under distribution.

    input:
        distribution: `torch.distributions.Distribution` of batch_shape either
            [batch_size, num_particles, dim1, ..., dimN] or
            [batch_size, dim1, ..., dimN] or
            [dim1, ..., dimN] or `dict` thereof.
        value: `torch.Tensor` of size
            [batch_size, num_particles, dim1, ..., dimN] +
            distribution.event_shape or `dict` thereof
        non_reparam: bool; if True, only returns log probability of the
            non-reparameterizable part of the distribution (default: False).

    output: `torch.Tensor` [batch_size, num_particles] or `dict` thereof
    """
    if isinstance(distribution, dict):
        return torch.sum(torch.cat([
            log_prob(v, value[k], non_reparam).unsqueeze(0)
            for k, v in distribution.items()
        ], dim=0), dim=0)
    elif isinstance(distribution, torch.distributions.Distribution):
        if non_reparam and distribution.has_rsample:
            return value.new(*value.size()[:2]).zero_()
        else:
            # non_reparam is True and distribution.has_rsample is False
            # or
            # non_reparam is False
            value_ndimension = value.ndimension()
            batch_shape_ndimension = len(distribution.batch_shape)
            event_shape_ndimension = len(distribution.event_shape)
            value_batch_shape_ndimension = \
                value_ndimension - event_shape_ndimension
            if (
                (value_batch_shape_ndimension == batch_shape_ndimension) or
                ((value_batch_shape_ndimension - 2) == batch_shape_ndimension)
            ):
                distribution._validate_sample(value)
                logp = distribution.log_prob(value)
            elif (value_batch_shape_ndimension - 1) == batch_shape_ndimension:
                logp = distribution.log_prob(
                    value.transpose(0, 1)).transpose(0, 1)
            else:
                raise RuntimeError(
                    'Incompatible distribution.batch_shape ({}) and '
                    'value.shape ({}).'.format(
                        distribution.batch_shape, value.shape
                    )
                )
            return torch.sum(logp.view(
                value.size(0), value.size(1), -1
            ), dim=2)
    else:
        raise AttributeError(
            'distribution must be a dict or a torch.distributions.Distribution.\
            Got: {}'.format(distribution)
        )


def resample(value, ancestral_index):
    """Resample the value without side effects.

    input:
        value: `torch.Tensor` [batch_size, num_particles, dim_1, ..., dim_N]
            (or [batch_size, num_particles]) or `dict` thereof
        ancestral_index: `torch.LongTensor` [batch_size, num_particles]
    output: resampled value [batch_size, num_particles, dim_1, ..., dim_N]
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
    """input:
        observation: `torch.Tensor` [batch_size, dim1, ..., dimN] or `dict`
            thereof
        num_particles: int

    output: `torch.Tensor` [batch_size, num_particles, dim1, ..., dimN] or
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

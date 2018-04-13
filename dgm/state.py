import torch


def set_expanded(distribution, expanded):
    """Sets the \"expanded\" property of a distribution explicitly."""

    distribution.expanded = expanded
    return distribution


def is_expanded(distribution, batch_size=None, num_particles=None):
    """Returns the \"expanded\" property of a distribution. If the property is
    not set explicitly, it is inferred implicitly by inspecting whether the
    first two dimensions of the distribution's batch_shape match
    (batch_size, num_particles).
    """

    if hasattr(distribution, 'expanded'):
        return distribution.expanded
    else:
        assert(isinstance(batch_size, int) and isinstance(num_particles, int))
        return distribution.batch_shape[:2] == (batch_size, num_particles)


def sample(distribution, batch_size, num_particles):
    """Samples from a distribution given batch size and number of particles.

    input:
        distribution: `torch.distributions.Distribution` or `dict` thereof.
            distribution can be either \"expanded\" in which case its
            batch_shape is [batch_size, num_particles, dim1, ..., dimN] or not
            \"expanded\" in which case its batch_shape is [dim1, ..., dimN].

            The \"expanded\" property can be set explicitly using the
            `set_expanded` function or it can be inferred implicitly. It is
            obtained using the `is_expanded` function.
        batch_size: `int`
        num_particles: `int`

    output: `torch.Tensor` [batch_size, num_particles, dim1, ..., dimN] or
        `dict` thereof
    """

    if isinstance(distribution, dict):
        return {
            k: sample(v, batch_size, num_particles)
            for k, v in distribution.items()
        }
    elif isinstance(distribution, torch.distributions.Distribution):
        if is_expanded(distribution, batch_size, num_particles):
            sample_shape = ()
        else:
            sample_shape = (batch_size, num_particles,)

        if distribution.has_rsample:
            return distribution.rsample(sample_shape=sample_shape)
        else:
            return distribution.sample(sample_shape=sample_shape)
    else:
        raise AttributeError(
            'distribution must be a dict or a torch.distributions.Distribution.\
            Got: {}'.format(distribution)
        )


def log_prob(distribution, value, non_reparam=False):
    """Log probability of value under distribution.

    input:
        distribution: `torch.distributions.Distribution` of batch_shape either
            [batch_size, num_particles, dim1, ..., dimN] or
            [dim1, ..., dimN] or `dict` thereof.
        value: `torch.Tensor` [batch_size, num_particles, dim1, ..., dimN] or
            `dict` thereof
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
            return torch.sum(distribution.log_prob(value).view(
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
        return {
            k: resample(v, ancestral_index)
            for k, v in value.items()
        }
    elif torch.is_tensor(value):
        assert(ancestral_index.size() == value.size()[:2])
        ancestral_index_unsqueezed = ancestral_index

        for _ in range(len(value.size()) - 2):
            ancestral_index_unsqueezed = \
                ancestral_index_unsqueezed.unsqueeze(-1)

        return torch.gather(
            value,
            dim=1,
            index=ancestral_index_unsqueezed.expand_as(value)
        )
    else:
        raise AttributeError(
            'value must be a dict or a torch.Tensor.\
            Got: {}'.format(value)
        )

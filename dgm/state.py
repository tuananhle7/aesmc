import torch


def set_expanded(distribution, expanded):
    distribution.expanded = expanded
    return distribution


def is_expanded(distribution, batch_size=None, num_particles=None):
    if hasattr(distribution, 'expanded'):
        return distribution.expanded
    else:
        assert(isinstance(batch_size, int) and isinstance(num_particles, int))
        return distribution.batch_shape[:2] == (batch_size, num_particles)


def sample(
    distribution, batch_size, num_particles, reparameterized
):
    if isinstance(distribution, dict):
        return {
            k: sample(v, batch_size, num_particles, reparameterized)
            for k, v in distribution.items()
        }
    elif isinstance(distribution, torch.distributions.Distribution):
        if is_expanded(distribution, batch_size, num_particles):
            sample_shape = ()
        else:
            sample_shape = (batch_size, num_particles,)

        if reparameterized:
            return distribution.rsample(
                sample_shape=sample_shape
            )
        else:
            return distribution.sample(
                sample_shape=sample_shape
            )
    else:
        raise AttributeError(
            'distribution must be a dict or a torch.distributions.Distribution.\
            Got: {}'.format(distribution)
        )


def log_prob(distribution, value):
    if isinstance(distribution, dict):
        return torch.sum(torch.cat([
            log_prob(v, value[k]).unsqueeze(0)
            for k, v in distribution.items()
        ], dim=0), dim=0)
    elif isinstance(distribution, torch.distributions.Distribution):
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

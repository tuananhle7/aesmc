import numpy as np
import scipy.misc
import torch


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a torch.Tensor/np.ndarray.

    Args:
        values: torch.Tensor/np.ndarray [dim_1, ..., dim_N]
        dim: n
    Returns:
        result: torch.Tensor/np.ndarray [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =

                                 exp(values[i_1, ..., i_N])
            log( ------------------------------------------------------------ )
                    sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    if isinstance(values, np.ndarray):
        log_denominator = scipy.special.logsumexp(
            values, axis=dim, keepdims=True
        )
        # log_numerator = values
        return values - log_denominator
    else:
        log_denominator = torch.logsumexp(values, dim=dim, keepdim=True)
        # log_numerator = values
        return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a torch.Tensor/np.ndarray.

    Args:
        values: torch.Tensor/np.ndarray [dim_1, ..., dim_N]
        dim: n
    Returns:
        result: torch.Tensor/np.ndarray [dim_1, ..., dim_N]
            where result[i_1, ..., i_N] =

                            exp(values[i_1, ..., i_N])
            ------------------------------------------------------------
             sum_{j = 1}^{dim_n} exp(values[i_1, ..., j, ..., i_N])
    """

    if isinstance(values, np.ndarray):
        return np.exp(lognormexp(values, dim=dim))
    else:
        return torch.exp(lognormexp(values, dim=dim))

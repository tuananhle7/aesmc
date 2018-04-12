import numpy as np
import scipy.misc
import torch


def logsumexp(values, dim=0, keepdim=False):
    """Logsumexp of a torch.Tensor.

    See https://en.wikipedia.org/wiki/LogSumExp.

    input:
        values: torch.Tensor [dim_1, ..., dim_N]
        dim: n

    output: result torch.Tensor
        [dim_1, ..., dim_{n - 1}, dim_{n + 1}, ..., dim_N] where

        result[i_1, ..., i_{n - 1}, i_{n + 1}, ..., i_N] =
            log(sum_{i_n = 1}^N exp(values[i_1, ..., i_N]))
    """

    values_max, _ = torch.max(values, dim=dim, keepdim=True)
    result = values_max + torch.log(torch.sum(
        torch.exp(values - values_max), dim=dim, keepdim=True
    ))
    return result if keepdim else result.squeeze(dim)


def lognormexp(values, dim=0):
    """Exponentiates, normalizes and takes log of a torch.Tensor/np.ndarray.

    input:
        values: torch.Tensor/np.ndarray [dim_1, ..., dim_N]
        dim: n
    output:
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
        log_denominator = logsumexp(values, dim=dim, keepdim=True)
        # log_numerator = values
        return values - log_denominator


def exponentiate_and_normalize(values, dim=0):
    """Exponentiates and normalizes a torch.Tensor/np.ndarray.

    input:
        values: torch.Tensor/np.ndarray [dim_1, ..., dim_N]
        dim: n
    output:
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

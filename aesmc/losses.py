from . import inference
import torch


def get_loss(observations, num_particles, algorithm, initial, transition,
             emission, proposal):
    """Returns a differentiable loss for gradient descent.

    Args:
        observations: list of tensors [batch_size, dim1, ..., dimN] or
            dicts thereof
        num_particles: int
        algorithm: iwae or aesmc
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
        proposal: a callable object (function or nn.Module) with signature:
            Args:
                previous_latents: list of length time where each element is a
                    tensor [batch_size, num_particles, ...]
                time: int
                observations: list where each element is a tensor
                    [batch_size, ...] or a dict thereof
            Returns: torch.distributions.Distribution or a dict thereof

    Returns:
        loss: scalar that we call .backward() on and step the optimizer.
    """
    if algorithm == 'iwae':
        inference_algorithm = 'is'
    elif algorithm == 'aesmc':
        inference_algorithm = 'smc'

    inference_result = inference.infer(inference_algorithm=inference_algorithm,
                                       observations=observations,
                                       initial=initial,
                                       transition=transition,
                                       emission=emission,
                                       proposal=proposal,
                                       num_particles=num_particles,
                                       return_log_marginal_likelihood=True,
                                       return_latents=False,
                                       return_original_latents=False,
                                       return_log_weight=False,
                                       return_log_weights=False,
                                       return_ancestral_indices=False)
    elbo = inference_result['log_marginal_likelihood']
    loss = -torch.mean(elbo)
    return loss

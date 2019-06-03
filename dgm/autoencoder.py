from . import inference

import enum
import torch
import torch.nn as nn


class AutoencoderAlgorithm(enum.Enum):
    VAE = 0  # variational autoencoder (IWAE with 1 particle)
    IWAE = 1  # importance weighted autoencoder
    AESMC = 2  # auto-encoding sequential monte carlo


class AutoEncoder(nn.Module):
    def __init__(self, initial, transition, emission, proposal):
        """Initialize AutoEncoder object.

        Args:
            initial: a callable object (function or nn.Module) which has no
                arguments and returns a torch.distributions.Distribution or a
                dict thereof
            transition: a callable object (function or nn.Module) with
                signature:
                Args:
                    previous_latent: tensor [batch_size, num_particles, ...]
                    time: int
                Returns: torch.distributions.Distribution or a dict thereof
            emission: a callable object (function or nn.Module) with signature:
                Args:
                    latent: tensor [batch_size, num_particles, ...]
                    time: int
                Returns: torch.distributions.Distribution or a dict thereof
            proposal: a callable object (function or nn.Module) with signature:
                Args:
                    previous_latent: tensor [batch_size, num_particles, ...]
                    time: int
                    observations: list where each element is a tensor
                        [batch_size, ...] or a dict thereof
                Returns: torch.distributions.Distribution or a dict thereof
        """
        super(AutoEncoder, self).__init__()
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.proposal = proposal

    def forward(self, observations, num_particles=2,
                autoencoder_algorithm=AutoencoderAlgorithm.IWAE):
        """Evaluate a computation graph whose gradient is an estimator for the
        gradient of the ELBO.

        Args:
            observations: list of tensors [batch_size, dim1, ..., dimN] or
                dicts thereof
            num_particles: int
            autoencoder_algorithm: AutoencoderAlgorithm value (default:
                AutoencoderAlgorithm.IWAE)

        Returns: tensor [batch_size]
        """

        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        if autoencoder_algorithm == AutoencoderAlgorithm.VAE:
            autoencoder_algorithm = AutoencoderAlgorithm.IWAE
            num_particles = 1

        if autoencoder_algorithm == AutoencoderAlgorithm.AESMC:
            inference_algorithm = inference.InferenceAlgorithm.SMC
        elif autoencoder_algorithm == AutoencoderAlgorithm.IWAE:
            inference_algorithm = inference.InferenceAlgorithm.IS

        inference_result = inference.infer(
            inference_algorithm=inference_algorithm,
            observations=observations,
            initial=self.initial,
            transition=self.transition,
            emission=self.emission,
            proposal=self.proposal,
            num_particles=num_particles,
            return_log_marginal_likelihood=True,
            return_latents=False,
            return_original_latents=False,
            return_log_weight=False,
            return_log_weights=False,
            return_ancestral_indices=False
        )
        elbo = inference_result['log_marginal_likelihood']

        return elbo

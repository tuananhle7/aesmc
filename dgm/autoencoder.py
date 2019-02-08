from . import inference
from . import math
from . import state

import enum
import numpy as np
import torch
import torch.nn as nn


class AutoencoderAlgorithm(enum.Enum):
    VAE = 0  # variational autoencoder (IWAE with 1 particle)
    IWAE = 1  # importance weighted autoencoder
    AESMC = 2  # auto-encoding sequential monte carlo


class AutoEncoder(nn.Module):
    def __init__(self, initial, transition, emission, proposal):
        """Initialize AutoEncoder object.

        input:
            initial: dgm.model.InitialDistribution object
            transition: dgm.model.TransitionDistribution object
            emission: dgm.model.EmissionDistribution object
            proposal: dgm.model.ProposalDistribution object
        """
        super(AutoEncoder, self).__init__()
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.proposal = proposal

    def forward(
        self,
        observations,
        num_particles=2,
        autoencoder_algorithm=AutoencoderAlgorithm.IWAE
    ):
        """Evaluate a computation graph whose gradient is an estimator for the
        gradient of the ELBO.

        input:
            observations: list of `torch.Tensor`s [batch_size, dim1, ..., dimN]
                or `dict`s thereof
            num_particles: int
            autoencoder_algorithm: AutoencoderAlgorithm value (default:
                AutoencoderAlgorithm.IWAE)

        output: torch.Tensor [batch_size]
        """

        for value, value_type, value_name, value_type_name in [
            [autoencoder_algorithm, AutoencoderAlgorithm,
             'autoencoder_algorithm', 'AutoencoderAlgorithm']
        ]:
            if not isinstance(value, value_type):
                raise TypeError('{} must be an {} enum.'.format(
                    value_name, value_type_name))

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

        # estimator_latents = inference_result['original_latents']  \
        #                     if inference_algorithm == inference.InferenceAlgorithm.SMC \
        #                     else inference_result['latents']
        # estimator = (inference.latents_log_prob(
        #     self.proposal,
        #     observations,
        #     estimator_latents,
        #     inference_result['ancestral_indices'],
        #     non_reparam=True
        # ) + inference.ancestral_indices_log_prob(
        #     inference_result['ancestral_indices'],
        #     inference_result['log_weights']
        # )) * inference_result['log_marginal_likelihood'].detach()

        return elbo

from . import inference
from . import math
from . import state
from . import wake_sleep

import enum
import numpy as np
import torch
import torch.nn as nn


class AutoencoderAlgorithm(enum.Enum):
    VAE = 0  # variational autoencoder (IWAE with 1 particle)
    IWAE = 1  # importance weighted autoencoder
    AESMC = 2  # auto-encoding sequential monte carlo
    WAKE_THETA = 3 
    SLEEP_PHI = 4 
    WAKE_PHI = 5  


class DiscreteGradientEstimator(enum.Enum):
    REINFORCE = 0
    VIMCO = 1


class ResamplingGradientEstimator(enum.Enum):
    IGNORE = 0
    REINFORCE = 1


class AutoEncoder(nn.Module):
    def __init__(self, initial, transition, emission, proposal):
        super(AutoEncoder, self).__init__()
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.proposal = proposal

    def forward(
        self,
        observations,
        num_particles=2,
        autoencoder_algorithm=AutoencoderAlgorithm.IWAE,
        wake_sleep_mode=wake_sleep.WakeSleepAlgorithm.IGNORE,
        discrete_gradient_estimator=DiscreteGradientEstimator.REINFORCE,
        resampling_gradient_estimator=ResamplingGradientEstimator.IGNORE
    ):
        """Evaluate a computation graph whose gradient is an estimator for the
        gradient of the ELBO.

        input:
            observations: list of `torch.Tensor`s [batch_size, dim1, ..., dimN]
                or `dict`s thereof
            num_particles: int
            autoencoder_algorithm: AutoencoderAlgorithm value (default:
                AutoencoderAlgorithm.IWAE)
            wake_sleep_algorightm: Wake Sleep estimator mode - specify if algorithm type is wake sleep
                (default: WakeSleepAlgorithm.IGNORE)
            discrete_gradient_estimator: DiscreteGradientEstimator value
                (default: DiscreteGradientEstimator.REINFORCE)
            resampling_gradient_estimator: ResamplingGradientEstimator value
                (default: ResamplingGradientEstimator.IGNORE)

        output: torch.Tensor [batch_size]
        """

        for value, value_type, value_name, value_type_name in [
            [autoencoder_algorithm, AutoencoderAlgorithm,
             'autoencoder_algorithm', 'AutoencoderAlgorithm'],
            [discrete_gradient_estimator, DiscreteGradientEstimator,
             'discrete_gradient_estimator', 'DiscreteGradientEstimator'],
            [resampling_gradient_estimator, ResamplingGradientEstimator,
             'resampling_gradient_estimator', 'ResamplingGradientEstimator']
        ]:
            if not isinstance(value, value_type):
                raise TypeError('{} must be an {} enum.'.format(
                    value_name, value_type_name
                ))

        batch_size = next(iter(observations[0].values())).size(0) \
            if isinstance(observations[0], dict) else observations[0].size(0)

        if autoencoder_algorithm == AutoencoderAlgorithm.VAE:
            autoencoder_algorithm = AutoencoderAlgorithm.IWAE
            num_particles = 1

        #  if autoencoder_algorithm == AutoencoderAlgorithm.WAKE_THETA:
        #      autoencoder_algorithm = AutoencoderAlgorithm.IWAE

        # TODO: implement
        # - DiscreteGradientEstimator.REINFORCE and VIMCO
        # - AutoencoderAlgorithm.WAKE_PHI, SLEEP_PHI


        #  if autoencoder_algorithm == AutoencoderAlgorithm.AESMC:
        if autoencoder_algorithm == AutoencoderAlgorithm.AESMC:
            if (
                resampling_gradient_estimator ==
                ResamplingGradientEstimator.IGNORE
            ):
                return inference.infer(
                    inference_algorithm=inference.InferenceAlgorithm.SMC,
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
                )['log_marginal_likelihood']
            elif (
                resampling_gradient_estimator ==
                ResamplingGradientEstimator.REINFORCE
            ):
                inference_result = inference.infer(
                    inference_algorithm=inference.InferenceAlgorithm.SMC,
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
                    return_log_weights=True,
                    return_ancestral_indices=True
                )
                log_ancestral_indices_proposal_ = \
                    inference.log_ancestral_indices_proposal(
                        inference_result['ancestral_indices'],
                        inference_result['log_weights']
                    )

                return log_ancestral_indices_proposal_ * \
                    inference_result['log_marginal_likelihood'].detach() + \
                    inference_result['log_marginal_likelihood']
            else:
                raise NotImplementedError('resampling_gradient_estimator {} \
                not implemented.'.format(resampling_gradient_estimator))
        elif autoencoder_algorithm == AutoencoderAlgorithm.IWAE:
            return inference.infer(
                inference_algorithm=inference.InferenceAlgorithm.IS,
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
            )['log_marginal_likelihood']

        elif autoencoder_algorithm == AutoencoderAlgorithm.WAKE_THETA:
            #  iwae_elbo =  inference.infer(
            #      inference_algorithm=inference.InferenceAlgorithm.IS,
            #      observations=observations,
            #      initial=self.initial,
            #      transition=self.transition,
            #      emission=self.emission,
            #      proposal=self.proposal,
            #      num_particles=num_particles,
            #      return_log_marginal_likelihood=True,
            #      return_latents=False,
            #      return_original_latents=False,
            #      return_log_weight=False,
            #      return_log_weights=False,
            #      return_ancestral_indices=False
            #  )['log_marginal_likelihood']
            #
            #  print("IWAE ELBO IS, ", iwae_elbo)
            ws_elbo = wake_sleep.infer(
                wake_sleep_mode=wake_sleep_mode,
                observations=observations,
                initial=self.initial,
                transition=self.transition,
                emission=self.emission,
                proposal=self.proposal,
                num_particles=num_particles,
                return_log_marginal_likelihood=True,
                return_latents=False,
                return_log_weight=False,
                return_log_weights=False
            )['log_marginal_likelihood']

            return ws_elbo
        elif autoencoder_algorithm == AutoencoderAlgorithm.SLEEP_PHI:
            return wake_sleep.infer(
                wake_sleep_mode=wake_sleep_mode,
                observations=observations,
                initial=self.initial,
                transition=self.transition,
                emission=self.emission,
                proposal=self.proposal,
                num_particles=num_particles,
                return_log_marginal_likelihood=True,
                return_latents=False,
                return_log_weight=False,
                return_log_weights=False
            )['log_marginal_likelihood']
        elif autoencoder_algorithm == AutoencoderAlgorithm.WAKE_PHI:
            return wake_sleep.infer(
                wake_sleep_mode=wake_sleep_mode,
                observations=observations,
                initial=self.initial,
                transition=self.transition,
                emission=self.emission,
                proposal=self.proposal,
                num_particles=num_particles,
                return_log_marginal_likelihood=True,
                return_latents=False,
                return_log_weight=False,
                return_log_weights=False
            )['log_marginal_likelihood']
        else:
            raise NotImplementedError('autoencoder_algorithm {} not \
            implemented.'.format(autoencoder_algorithm))

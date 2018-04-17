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
    VIMCO = 3
    WAKE_SLEEP = 4 

class DiscreteGradientEstimator(enum.Enum):
    IGNORE = 0
    REINFORCE = 1
    VIMCO = 2


class ResamplingGradientEstimator(enum.Enum):
    IGNORE = 0
    REINFORCE = 1

class WakeSleepAlgorithm(enum.Enum):
    IGNORE = 0
    WS = 1  
    WW = 2
    WSW = 3 
    WSWA = 4

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
        autoencoder_algorithm=AutoencoderAlgorithm.IWAE,
        discrete_gradient_estimator=DiscreteGradientEstimator.REINFORCE,
        resampling_gradient_estimator=ResamplingGradientEstimator.IGNORE,
        wake_sleep_mode=WakeSleepAlgorithm.IGNORE,
        wake_optimizer=None,
        sleep_optimizer=None
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

        return_log_marginal_likelihood = True
        return_latents = False
        return_original_latents = False
        return_log_weight = False
        return_log_weights = False
        return_ancestral_indices=False

        if autoencoder_algorithm == AutoencoderAlgorithm.AESMC:
            """
            gradient and resampling estimators are incorporated into the loss approprietely below
            """
            inference_algorithm = inference.InferenceAlgorithm.SMC
            if resampling_gradient_estimator == ResamplingGradientEstimator.IGNORE \
            and discrete_gradient_estimator == DiscreteGradientEstimator.REINFORCE:
                return_ancestral_indices = True
                return_original_latents = True
            elif resampling_gradient_estimator == ResamplingGradientEstimator.REINFORCE \
            and discrete_gradient_estimator == DiscreteGradientEstimator.IGNORE:
                return_log_weights = True 
                return_ancestral_indices = True
            elif resampling_gradient_estimator == ResamplingGradientEstimator.REINFORCE \
            and discrete_gradient_estimator == DiscreteGradientEstimator.REINFORCE:
                return_log_weights = True 
                return_ancestral_indices = True
                return_original_latents = True
            elif resampling_gradient_estimator == ResamplingGradientEstimator.IGNORE \
            and discrete_gradient_estimator == DiscreteGradientEstimator.IGNORE:
                pass
            else:
                raise NotImplementedError('cannot use  {}, {}, and {} together.'.format(\
                        discrete_gradient_estimator, resampling_gradient_estimaor, autoencoder_algorithm))
        elif autoencoder_algorithm == AutoencoderAlgorithm.IWAE:
            inference_algorithm = inference.InferenceAlgorithm.IS
            if discrete_gradient_estimator == DiscreteGradientEstimator.REINFORCE:
                return_latents = True
            if discrete_gradient_estimator == DiscreteGradientEstimator.VIMCO:
                return_latents = True
                return_log_weights = True
        elif autoencoder_algorithm == AutoencoderAlgorithm.WAKE_SLEEP:
            inference_algorithm = inference.InferenceAlgorithm.WS
            if wake_optimizer is None:
                raise NotImplementedError('cannot use wake sleep mode without optimizer for model parameters')
            #  if wake_sleep_mode == WakeSleepAlgorithm.IGNORE or wake_sleep_mode > WakeSleepAlgorithm.WW:
            if wake_sleep_mode == WakeSleepAlgorithm.IGNORE:
                raise NotImplementedError('cannot use wake sleep mode {} and {} together.'.format(\
                        wake_sleep_mode, autoencoder_algorithm))
        else:
            raise NotImplementedError('autoencoder_algorithm {} not \
            implemented.'.format(autoencoder_algorithm))

        inference_result = inference.infer(
            inference_algorithm=inference_algorithm,
            wake_sleep_mode=wake_sleep_mode,
            wake_optimizer=wake_optimizer,
            sleep_optimizer=sleep_optimizer,
            observations=observations,
            initial=self.initial,
            transition=self.transition,
            emission=self.emission,
            proposal=self.proposal,
            num_particles=num_particles,
            return_log_marginal_likelihood=return_log_marginal_likelihood,
            return_latents=return_latents,
            return_original_latents=return_original_latents,
            return_log_weight=return_log_weight,
            return_log_weights=return_log_weights,
            return_ancestral_indices=return_ancestral_indices
        )

        elbo = inference_result['log_marginal_likelihood']
        
        # uses vimco estimator with iwae if specified, otherwise estimator 
        # becomes the correct model/ancestral estimator or nothing
        if discrete_gradient_estimator == DiscreteGradientEstimator.VIMCO:
            estimator = inference.control_variate(
                self.proposal,
                observations,
                inference_result['log_marginal_likelihood'].detach(),
                num_particles,
                inference_result['log_latents'],
                inference_result['log_weights'],
                non_reparam=True
            ) 
        else:
            estimator_latents = inference_result['original_latents']  \
                                if inference_algorithm == inference.InferenceAlgorithm.SMC \
                                else inference_result['latents']
            estimator = (inference.latents_log_prob( 
                self.proposal,
                observations,
                estimator_latents,
                inference_result['ancestral_indices'],
                non_reparam=True
            ) + inference.ancestral_indices_log_prob(
                inference_result['ancestral_indices'],
                inference_result['log_weights']
            )) * inference_result['log_marginal_likelihood'].detach()
            
        return elbo + estimator


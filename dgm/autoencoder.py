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
    WAKE_THETA = 3  # wake update of theta in reweighted wake-sleep
    WAKE_PHI = 4  # wake update of phi in reweighted wake-sleep
    SLEEP_PHI = 5  # sleep update of phi in reweighted wake-sleep


class DiscreteGradientEstimator(enum.Enum):
    IGNORE = 0
    REINFORCE = 1
    VIMCO = 2


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

        if autoencoder_algorithm == AutoencoderAlgorithm.WAKE_THETA:
            autoencoder_algorithm = AutoencoderAlgorithm.IWAE
            discrete_gradient_estimator = DiscreteGradientEstimator.IGNORE

        # TODO: implement
        # - DiscreteGradientEstimator.VIMCO
        # - AutoencoderAlgorithm.WAKE_PHI, SLEEP_PHI
        if autoencoder_algorithm == AutoencoderAlgorithm.AESMC:
            if (
                resampling_gradient_estimator ==
                ResamplingGradientEstimator.IGNORE
            ):
                if (
                    discrete_gradient_estimator ==
                    DiscreteGradientEstimator.IGNORE
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
                    discrete_gradient_estimator ==
                    DiscreteGradientEstimator.REINFORCE
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
                        return_original_latents=True,
                        return_log_weight=False,
                        return_log_weights=False,
                        return_ancestral_indices=True
                    )
                    return inference_result['log_marginal_likelihood'] + \
                        inference_result['log_marginal_likelihood'].detach() *\
                        inference.latents_log_prob(
                            self.proposal,
                            observations,
                            inference_result['original_latents'],
                            inference_result['ancestral_indices'],
                            non_reparam=True
                        )
                elif (
                    discrete_gradient_estimator ==
                    DiscreteGradientEstimator.VIMCO
                ):
                    raise ValueError('{} and {} are not compatible'.format(
                        AutoencoderAlgorithm.AESMC,
                        DiscreteGradientEstimator.VIMCO
                    ))
                else:
                    raise NotImplementedError('discrete_gradient_estimator {} \
                    not implemented.'.format(discrete_gradient_estimator))
            elif (
                resampling_gradient_estimator ==
                ResamplingGradientEstimator.REINFORCE
            ):
                if (
                    discrete_gradient_estimator ==
                    DiscreteGradientEstimator.IGNORE
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
                    return inference_result['log_marginal_likelihood'] + \
                        inference_result['log_marginal_likelihood'].detach() *\
                        inference.ancestral_indices_log_prob(
                            inference_result['ancestral_indices'],
                            inference_result['log_weights']
                        )
                elif (
                    discrete_gradient_estimator ==
                    DiscreteGradientEstimator.REINFORCE
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
                        return_original_latents=True,
                        return_log_weight=False,
                        return_log_weights=True,
                        return_ancestral_indices=True
                    )
                    return inference_result['log_marginal_likelihood'] + \
                        inference_result['log_marginal_likelihood'].detach() *\
                        (
                            inference.latents_log_prob(
                                self.proposal,
                                observations,
                                inference_result['original_latents'],
                                inference_result['ancestral_indices'],
                                non_reparam=True
                            ) + inference.ancestral_indices_log_prob(
                                inference_result['ancestral_indices'],
                                inference_result['log_weights']
                            )
                        )
                elif (
                    discrete_gradient_estimator ==
                    DiscreteGradientEstimator.VIMCO
                ):
                    raise ValueError('{} and {} are not compatible'.format(
                        AutoencoderAlgorithm.AESMC,
                        DiscreteGradientEstimator.VIMCO
                    ))
                else:
                    raise NotImplementedError('discrete_gradient_estimator {} \
                     not implemented.'.format(discrete_gradient_estimator))
            else:
                raise NotImplementedError('resampling_gradient_estimator {} \
                not implemented.'.format(resampling_gradient_estimator))
        elif autoencoder_algorithm == AutoencoderAlgorithm.IWAE:
            if discrete_gradient_estimator == DiscreteGradientEstimator.IGNORE:
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
            elif (
                discrete_gradient_estimator ==
                DiscreteGradientEstimator.REINFORCE
            ):
                inference_result = inference.infer(
                    inference_algorithm=inference.InferenceAlgorithm.IS,
                    observations=observations,
                    initial=self.initial,
                    transition=self.transition,
                    emission=self.emission,
                    proposal=self.proposal,
                    num_particles=num_particles,
                    return_log_marginal_likelihood=True,
                    return_latents=True,
                    return_original_latents=False,
                    return_log_weight=False,
                    return_log_weights=False,
                    return_ancestral_indices=False
                )
                return inference_result['log_marginal_likelihood'] + \
                    inference_result['log_marginal_likelihood'].detach() * \
                    inference.latents_log_prob(
                        self.proposal,
                        observations,
                        inference_result['latents'],
                        non_reparam=True
                    )
            else:
                raise NotImplementedError('discrete_gradient_estimator {} not \
                implemented.'.format(discrete_gradient_estimator))
        else:
            raise NotImplementedError('autoencoder_algorithm {} not \
            implemented.'.format(autoencoder_algorithm))

from . import autoencoder as ae
from . import statistics
from . import inference
from . import state

import itertools
import sys
import torch.nn as nn
import torch.utils.data


#  def forward_generative_model(
#      autoencoder,
#      dataloader,
#      observations,
#      autoencoder_algorithm,
#      num_epochs,
#      num_iterations_per_epoch=None,
#      num_particles=None,
#      wake_sleep_mode=ae.WakeSleepAlgorithm.IGNORE,
#      discrete_gradient_estimator=ae.DiscreteGradientEstimator.REINFORCE,
#      resampling_gradient_estimator=ae.ResamplingGradientEstimator.IGNORE,
#      process_data_hook=None,
#      callback=None
#  ):
#      if autoencoder_algorithm in [
#          ae.AutoencoderAlgorithm.VAE,
#          ae.AutoencoderAlgorithm.IWAE,
#          ae.AutoencoderAlgorithm.AESMC,
#          ae.AutoencoderAlgorithm.WAKE_SLEEP
#      ]:
#          if process_data_hook is not None:
#              observations = process_data_hook(observations, autoencoder)
#
#          inference_result = inference.infer(
#              inference_algorithm=inference.InferenceAlgorithm.IS,
#              wake_sleep_mode=None,
#              wake_optimizer=None,
#              sleep_optimizer=None,
#              observations=observations,
#              initial=autoencoder.initial,
#              transition=autoencoder.transition,
#              emission=autoencoder.emission,
#              proposal=autoencoder.proposal,
#              num_particles=num_particles,
#              return_log_marginal_likelihood=True,
#              return_latents=True,
#              return_original_latents=False,
#              return_log_weight=False,
#              return_log_weights=False,
#              return_ancestral_indices=False
#          )
#          latent = inference_result['last_latent']
#          samples = []
#          for i in range(100):
#              _transition = autoencoder.transition.transition(previous_latent=latent, time=1)
#              next_latent = state.sample(_transition, 1, num_particles)
#              _emission = autoencoder.emission.emission(latent=next_latent, time=1)
#              next_emission = state.sample(_emission, 1, 1)
#              emissions = []
#              for key, value in next_emission.items():
#                  obs = torch.mean(next_emission[key], dim=1).unsqueeze(1)
#                  emissions.append(obs)
#              all_flies = torch.cat(emissions, dim=0)
#              samples.append(all_flies)
#              latent = next_latent
#          return torch.cat(samples, dim=1)
#
#      else:
#          raise NotImplementedError(
#              'autoencoder_algorithm {} not implemented.'.format(
#                  autoencoder_algorithm
#              )
#          )
def predict_next_obs(
    autoencoder,
    dataloader,
    observations,
    autoencoder_algorithm,
    num_epochs,
    num_iterations_per_epoch=None,
    num_particles=None,
    wake_sleep_mode=ae.WakeSleepAlgorithm.IGNORE,
    discrete_gradient_estimator=ae.DiscreteGradientEstimator.REINFORCE,
    resampling_gradient_estimator=ae.ResamplingGradientEstimator.IGNORE,
    process_data_hook=None,
    callback=None
):
    if autoencoder_algorithm in [
        ae.AutoencoderAlgorithm.VAE,
        ae.AutoencoderAlgorithm.IWAE,
        ae.AutoencoderAlgorithm.AESMC,
        ae.AutoencoderAlgorithm.WAKE_SLEEP
    ]:
        if process_data_hook is not None:
            observations = process_data_hook(observations, autoencoder)

        inference_result = inference.infer(
            inference_algorithm=inference.InferenceAlgorithm.IS,
            wake_sleep_mode=None,
            wake_optimizer=None,
            sleep_optimizer=None,
            observations=observations,
            initial=autoencoder.initial,
            transition=autoencoder.transition,
            emission=autoencoder.emission,
            proposal=autoencoder.proposal,
            num_particles=num_particles,
            return_log_marginal_likelihood=True,
            return_latents=True,
            return_original_latents=False,
            return_log_weight=False,
            return_log_weights=False,
            return_ancestral_indices=False
        )
        latent = inference_result['last_latent']
        _transition = autoencoder.transition.transition(previous_latent=latent, time=1)
        next_latent = state.sample(_transition, 1, num_particles)
        _emission = autoencoder.emission.emission(latent=next_latent, time=1)
        next_emission = state.sample(_emission, 1, num_particles)
        return next_emission

    else:
        raise NotImplementedError(
            'autoencoder_algorithm {} not implemented.'.format(
                autoencoder_algorithm
            )
        )

# TODO: implement tests (?)

from . import autoencoder as ae
from . import statistics

import itertools
import sys
import torch.nn as nn
import torch.utils.data


# TODO: Consider putting this as a method of the AutoEncoder object
def get_theta_parameters(autoencoder):
    result = []
    for network in [
        autoencoder.initial, autoencoder.transition, autoencoder.emission
    ]:
        if (network is not None) and isinstance(network, nn.Module):
            result = itertools.chain(result, network.parameters())

    if isinstance(result, list):
        return None
    else:
        return result


# TODO: Consider putting this as a method of the AutoEncoder object
def get_phi_parameters(autoencoder):
    if (
        (autoencoder.proposal is None) or
        (not isinstance(autoencoder.proposal, nn.Module))
    ):
        return None
    else:
        return autoencoder.proposal.parameters()


def train_autoencoder(
    autoencoder,
    dataloader,
    autoencoder_algorithm,
    num_epochs,
    num_iterations_per_epoch=None,
    num_particles=None,
    wake_sleep_mode=ae.WakeSleepAlgorithm.IGNORE,
    discrete_gradient_estimator=ae.DiscreteGradientEstimator.REINFORCE,
    resampling_gradient_estimator=ae.ResamplingGradientEstimator.IGNORE,
    optimizer_algorithm=torch.optim.Adam,
    optimizer_kwargs={},
    theta_optimizer_algorithm=None,
    theta_optimizer_kwargs=None,
    phi_optimizer_algorithm=None,
    phi_optimizer_kwargs=None,
    callback=None
):
    if theta_optimizer_algorithm is None:
        theta_optimizer_algorithm = optimizer_algorithm
        theta_optimizer_kwargs = optimizer_kwargs

    if phi_optimizer_algorithm is None:
        phi_optimizer_algorithm = optimizer_algorithm
        phi_optimizer_kwargs = optimizer_kwargs

    theta_parameters = get_theta_parameters(autoencoder)
    phi_parameters = get_phi_parameters(autoencoder)

    optimize_theta = theta_parameters is not None
    optimize_phi = phi_parameters is not None

    if optimize_theta:
        theta_optimizer = theta_optimizer_algorithm(
            theta_parameters, **theta_optimizer_kwargs
        )
    if optimize_phi:
        phi_optimizer = phi_optimizer_algorithm(
            phi_parameters, **phi_optimizer_kwargs
        )

    if autoencoder_algorithm in [
        ae.AutoencoderAlgorithm.VAE,
        ae.AutoencoderAlgorithm.IWAE,
        ae.AutoencoderAlgorithm.AESMC,
        ae.AutoencoderAlgorithm.WAKE_SLEEP
    ]:
        for epoch_idx in range(num_epochs):
            for epoch_iteration_idx, observations in enumerate(dataloader):
                if num_iterations_per_epoch is not None:
                    if epoch_iteration_idx == num_iterations_per_epoch:
                        break

                if optimize_theta:
                    theta_optimizer.zero_grad()

                if optimize_phi:
                    phi_optimizer.zero_grad()

                elbo = autoencoder.forward(
                    observations,
                    num_particles,
                    autoencoder_algorithm,
                    discrete_gradient_estimator,
                    resampling_gradient_estimator,
                    wake_sleep_mode,
                    theta_optimizer,
                    phi_optimizer
                )
                loss = -torch.mean(elbo)

                loss.backward()

                if optimize_theta and autoencoder_algorithm != ae.AutoencoderAlgorithm.WAKE_SLEEP:
                    theta_optimizer.step()

                if optimize_phi:
                    phi_optimizer.step()

                if callback is not None:
                    callback(epoch_idx, epoch_iteration_idx, autoencoder)
    else:
        raise NotImplementedError(
            'autoencoder_algorithm {} not implemented.'.format(
                autoencoder_algorithm
            )
        )


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(
        self, initial, transition, emission, num_timesteps, batch_size
    ):
        self.initial = initial
        self.transition = transition
        self.emission = emission
        self.num_timesteps = num_timesteps
        self.batch_size = batch_size

    def __getitem__(self, index):
        # TODO this is wrong, obs can be dict
        return list(map(
            lambda observation: observation.detach().squeeze(0),
            statistics.sample_from_prior(
                self.initial,
                self.transition,
                self.emission,
                self.num_timesteps,
                self.batch_size
            )[1]
        ))

    def __len__(self):
        return sys.maxsize  # effectively infinite


def get_synthetic_dataloader(
    initial, transition, emission, num_timesteps, batch_size
):
    return torch.utils.data.DataLoader(
        SyntheticDataset(
            initial, transition, emission, num_timesteps, batch_size
        ),
        batch_size=1,
        collate_fn=lambda x: x[0]
    )

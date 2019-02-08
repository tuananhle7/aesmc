import copy
import dgm
import numpy as np
import pykalman
import torch
import torch.nn as nn
import torch.nn.functional as F


class Initial(dgm.model.InitialDistribution):
    def __init__(self, loc, scale):
        super(Initial, self).__init__()
        self.loc = loc
        self.scale = scale

    def initial(self):
        return torch.distributions.Normal(self.loc, self.scale)


class Transition(dgm.model.TransitionNetwork):
    def __init__(self, init_mult, scale):
        super(Transition, self).__init__()
        self.mult = nn.Parameter(torch.Tensor([init_mult]).squeeze())
        self.scale = scale

    def transition(self, previous_latent=None, time=None):
        return dgm.state.set_batch_shape_mode(
            torch.distributions.Normal(
                self.mult * previous_latent, self.scale
            ),
            dgm.state.DistributionBatchShapeMode.FULLY_EXPANDED
        )


class Emission(dgm.model.EmissionNetwork):
    def __init__(self, init_mult, scale):
        super(Emission, self).__init__()
        self.mult = nn.Parameter(torch.Tensor([init_mult]).squeeze())
        self.scale = scale

    def emission(self, latent=None, time=None):
        return dgm.state.set_batch_shape_mode(
            torch.distributions.Normal(self.mult * latent, self.scale),
            dgm.state.DistributionBatchShapeMode.FULLY_EXPANDED
        )


class Proposal(dgm.model.ProposalNetwork):
    def __init__(self, scale_0, scale_t):
        super(Proposal, self).__init__()
        self.scale_0 = scale_0
        self.scale_t = scale_t
        self.lin_0 = nn.Linear(1, 1)
        self.lin_t = nn.Linear(2, 1)

    def proposal(self, previous_latent=None, time=None, observations=None):
        if time == 0:
            return dgm.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_0(observations[0].unsqueeze(-1)).squeeze(-1),
                    scale=self.scale_0
                ),
                dgm.state.DistributionBatchShapeMode.BATCH_EXPANDED
            )
        else:
            num_particles = previous_latent.shape[1]
            return dgm.state.set_batch_shape_mode(
                torch.distributions.Normal(
                    loc=self.lin_t(torch.cat(
                        [previous_latent.unsqueeze(-1),
                         observations[time].view(-1, 1, 1).expand(
                            -1, num_particles, 1
                         )],
                        dim=2
                    ).view(-1, 2)).squeeze(-1).view(-1, num_particles),
                    scale=self.scale_0
                ),
                dgm.state.DistributionBatchShapeMode.FULLY_EXPANDED
            )


def lgssm_true_posterior(
    observations,
    initial_loc,
    initial_scale,
    transition_mult,
    transition_bias,
    transition_scale,
    emission_mult,
    emission_bias,
    emission_scale
):
    kf = pykalman.KalmanFilter(
        initial_state_mean=[initial_loc],
        initial_state_covariance=[[initial_scale**2]],
        transition_matrices=[[transition_mult]],
        transition_offsets=[transition_bias],
        transition_covariance=[[transition_scale**2]],
        observation_matrices=[[emission_mult]],
        observation_offsets=[emission_bias],
        observation_covariance=[[emission_scale**2]]
    )

    return kf.smooth(observations)


class TrainingStats(object):
    def __init__(self, initial_loc, initial_scale, true_transition_mult,
                 transition_scale, true_emission_mult, emission_scale,
                 num_timesteps, num_test_obs, test_inference_num_particles,
                 saving_interval=100, logging_interval=100):
        self.true_transition_mult = true_transition_mult
        self.true_emission_mult = true_emission_mult
        self.test_inference_num_particles = test_inference_num_particles
        self.saving_interval = saving_interval
        self.logging_interval = logging_interval
        self.p_l2_history = []
        self.q_l2_history = []
        self.iteration_idx_history = []
        self.initial = Initial(initial_loc, initial_scale)
        self.true_transition = Transition(true_transition_mult,
                                          transition_scale)
        self.true_emission = Emission(true_emission_mult, emission_scale)
        dataloader = dgm.train.get_synthetic_dataloader(self.initial,
                                                        self.true_transition,
                                                        self.true_emission,
                                                        num_timesteps,
                                                        num_test_obs)
        self.test_obs = next(iter(dataloader))
        self.true_posterior_means = [None] * num_test_obs
        for test_obs_idx in range(num_test_obs):
            observations = [[o[test_obs_idx]] for o in self.test_obs]
            self.true_posterior_means[test_obs_idx] = np.reshape(
                lgssm_true_posterior(observations, initial_loc, initial_scale,
                                     self.true_transition_mult, 0,
                                     transition_scale, self.true_emission_mult,
                                     0, emission_scale)[0],
                (-1)
            )

    def __call__(self, epoch_idx, epoch_iteration_idx, elbo, loss, autoencoder):
        if epoch_iteration_idx % self.saving_interval == 0:
            self.p_l2_history.append(np.linalg.norm(
                np.array([autoencoder.transition.mult.item(),
                          autoencoder.emission.mult.item()]) -
                np.array([self.true_transition_mult,
                          self.true_emission_mult])
            ))
            inference_result = dgm.inference.infer(
                dgm.inference.InferenceAlgorithm.IS,
                self.test_obs, self.initial, self.true_transition,
                self.true_emission, autoencoder.proposal,
                self.test_inference_num_particles
            )
            posterior_means = dgm.statistics.empirical_mean(
                torch.cat([latent.unsqueeze(-1)
                           for latent in inference_result['latents']],
                          dim=2),
                inference_result['log_weight']
            ).detach().numpy()
            # import pdb; pdb.set_trace()
            self.q_l2_history.append(np.mean(np.linalg.norm(
                self.true_posterior_means - posterior_means, axis=1
            )))
            self.iteration_idx_history.append(epoch_iteration_idx)

        if epoch_iteration_idx % self.logging_interval == 0:
            print('Iteration: {}'.format(epoch_iteration_idx))

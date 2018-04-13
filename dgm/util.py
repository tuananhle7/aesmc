from . import math
from . import state

import numpy as np
import torch

def expand_observation(observation, num_particles):
    """input:
        observation: `torch.Tensor` [batch_size, dim1, ..., dimN]
        num_particles: int

    output: `torch.Tensor` [batch_size, num_particles, dim1, ..., dimN]
    """
    batch_size = observation.size(0)
    other_sizes = list(observation.size()[1:])

    return observation.unsqueeze(1).expand(
        *([batch_size, num_particles] + other_sizes)
    )

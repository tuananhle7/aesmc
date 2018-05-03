import torch.nn as nn


class InitialDistribution():
    """Base class for initial distribution"""

    def initial(self):
        """Returns `torch.distributions.Distribution` or a `dict` thereof."""
        raise NotImplementedError


class InitialNetwork(nn.Module, InitialDistribution):
    """Base class for initial networks.

    Your initial networks should subclass this class.

    You must override __init__() and initial(self).

    Your `__init__` class must call `__init__` of the superclass.

    E.g. TODO

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call .cuda(), etc.
    """

    def __init__(self):
        super(InitialNetwork, self).__init__()


class TransitionDistribution():
    """Base class for transition distribution"""

    def transition(self, previous_latent=None, sampled_latent=None, time=None):
        """Returns `torch.distributions.Distribution` or a `dict` thereof."""
        raise NotImplementedError


class TransitionNetwork(nn.Module, TransitionDistribution):
    """Base class for transition networks.

    Your transition networks should subclass this class.

    You must override __init__() and
    transition(self, previous_latent=None).

    Your `__init__` class must call `__init__` of the superclass.

    E.g. TODO

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call .cuda(), etc.
    """

    def __init__(self, *args, **kwargs):
        super(TransitionNetwork, self).__init__(*args, **kwargs)


class EmissionDistribution():
    """Base class for emission distribution"""

    def emission(self, latent=None, time=None):
        """Returns `torch.distributions.Distribution` or a `dict` thereof."""
        raise NotImplementedError


class EmissionNetwork(nn.Module, EmissionDistribution):
    """Base class for emission networks.

    Your emission networks should subclass this class.

    You must override __init__() and emission(latent=None).

    Your `__init__` class must call `__init__` of the superclass.

    E.g. TODO

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call .cuda(), etc.
    """

    def __init__(self):
        super(EmissionNetwork, self).__init__()


class ProposalDistribution():
    """Base class for proposal distribution"""

    def proposal(
        self, previous_latent=None, time=None, observations=None
    ):
        """Returns `torch.distributions.Distribution` or a `dict` thereof."""
        raise NotImplementedError


class ProposalNetwork(nn.Module, ProposalDistribution):
    """Base class for proposal networks.

    Your proposal networks should subclass this class.

    You must override __init__() and
    proposal(previous_latent=None, time=None).

    Your `__init__` class must call `__init__` of the superclass.

    E.g. TODO

    Submodules assigned in this way will be registered, and will have their
    parameters converted too when you call .cuda(), etc.
    """

    def __init__(self):
        super(ProposalNetwork, self).__init__()

import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class AbstractNoiseModel(ABC):
    """ Abstract class. All noise models inherit from here.
    """

    def __init__(self, *argv, **kwargs):
        """ Initialize an AbstractNoiseModel object.
        """

    @abc.abstractmethod
    def loss(self, *argv, **kwargs):
        """ Compute loss given predictions and groundtruth labels
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_noise_var(self, *argv, **kwargs):
        """
        Return the current estimate of noise variance
        """
        raise NotImplementedError

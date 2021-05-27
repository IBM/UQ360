import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class BuiltinUQ(ABC):
    """ BuiltinUQ is the base class for any algorithm that has UQ built into it.
    """

    def __init__(self, *argv, **kwargs):
        """ Initialize a BuiltinUQ object.
        """

    @abc.abstractmethod
    def fit(self, *argv, **kwargs):
        """ Learn the UQ related parameters..
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, *argv, **kwargs):
        """ Method to obtain the predicitve uncertainty, this can return the total, epistemic and/or aleatoric
         uncertainty in the predictions.
        """
        raise NotImplementedError

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self



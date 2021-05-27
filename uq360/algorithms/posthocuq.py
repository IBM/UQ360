import abc
import sys

# Ensure compatibility with Python 2/3
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class PostHocUQ(ABC):
    """ PostHocUQ is the base class for any algorithm that quantifies uncertainty of a pre-trained model.
    """

    def __init__(self, *argv, **kwargs):
        """ Initialize a BuiltinUQ object.
        """

    @abc.abstractmethod
    def _process_pretrained_model(self, *argv, **kwargs):
        """ Method to process the pretrained model that requires UQ.
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

    def get_params(self):
        """
         This method should not take any arguments and returns a dict of the __init__ parameters.

        """
        raise NotImplementedError

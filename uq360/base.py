
import logging
from typing import Any, Dict, List, Optional, Union, Tuple 


class Base(object):
    """ Base class that represents a generic plugin with subtypes that are
        referenced by name and instantiated at runtime. """

    @classmethod
    def name(cls) -> Union[str, List[str]]:
        """ Name of this subtype, used for lookup purposes. To be implemented by subclasses.
            This method can either return a single name, or a list/tuple of names. """
        raise NotImplementedError()

    @classmethod
    def instance(cls, discriminator: Any, **params) -> 'PluginBase':
        return cls._instance(discriminator, **params)

    @classmethod
    def _instance(cls, discriminator: Any, **params) -> 'PluginBase':
        for subclazz in cls.__subclasses__():
            inst = subclazz.try_instantiate(discriminator, **params)
            if inst:
                return inst
            # recursive call for subclass
            try:
                return subclazz._instance(discriminator, **params)
            except Exception as e:
                if not isinstance(e, NotImplementedError):
                    logging.warning('Unable to instantiate %s subtype "%s": %s' % (subclazz.__name__, discriminator, e))
                    print(e)

        raise NotImplementedError('Unable to find or create %s type "%s"' % (cls.__name__, discriminator))

    @classmethod
    def try_instantiate(cls, discriminator: Any, **params) -> Optional['PluginBase']:
        """ Return an instance of this subtype, if it is capable of handling the given discriminator, otherwise None.
            By default, this method assumes that "discriminator" is a subtype name, and returns an instance if
            the discriminator matches the name of this subtype (as returned by the `name()` method).
            This method can be overwritten by subclasses, in which case the discriminator can also be any
            other object based on which the subtype class can decide whether it is responsible
            for handling it. """
        try:
            name = cls.name()
        except Exception:
            name = None
        if name == discriminator or (is_list_or_tuple(name) and discriminator in name):
            return cls(**params)


def is_list_or_tuple(obj):
    """ Determine whether an object is an iterable list or tuple """
    return isinstance(obj, (list, tuple))

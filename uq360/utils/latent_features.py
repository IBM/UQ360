from typing import Callable, List, Union

from torch import no_grad
from torch.nn import Module


class LatentFeatures:
    def __init__(
        self,
        model: Callable,
        layer: Union[Module, List[Module]],
        post_processing_fn=None,
        out_device: str = "cpu",
    ):

        self.model = model
        self.layer = layer if isinstance(layer, list) else [layer]
        self.post_processing_fn = post_processing_fn
        self.out_device = out_device

    def extract(self, input):

        activations = []
        hooks = []

        for single_layer in self.layer:
            hook_fn = lambda m, i, output: activations.append(output)
            hook = single_layer.register_forward_hook(hook_fn)
            hooks.append(hook)

        with no_grad():
            self.model(input)

        for hook in hooks:
            hook.remove()

        if self.post_processing_fn:
            activations = [
                self.post_processing_fn(layer_act) for layer_act in activations
            ]

        activations = [a.to(self.out_device) for a in activations]

        return activations

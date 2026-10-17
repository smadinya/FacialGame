import numpy as np
from models.model import Model
from models.transfered_model import load_trunk, attach_head, compile_head


class RandomModel(Model):
    """No-transfer baseline: same architecture and same trainable head as TransferedModel,
    but the trunk's learned weights are thrown away first. Only the trunk differs."""

    def _define_model(self, input_shape, categories_count):
        trunk = load_trunk()
        self._randomize_layers(trunk)
        self.model = attach_head(trunk, categories_count)

    def _compile_model(self):
        compile_head(self.model)

    @staticmethod
    def _randomize_layers(model):
        for layer in model.layers:
            weights = layer.get_weights()
            if weights:
                # ponytail: fixed-scale normal rather than each layer's own initializer.
                # 0.05 is close to glorot for these shapes; swap in layer.kernel_initializer
                # if the baseline turns out to be handicapped rather than merely untrained.
                layer.set_weights([np.random.normal(0., 0.05, w.shape) for w in weights])

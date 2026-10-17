from models.model import Model
from tensorflow.keras import Input, layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import RMSprop
from config import transfer_base_model, transfer_feature_layer


def load_trunk():
    """The section 6 model truncated to its feature layer, frozen.

    Defaults to dropping only the final softmax (config.transfer_feature_layer = None).
    Set that to a layer name to tap earlier, generic features instead.
    """
    base = load_model(transfer_base_model)
    if transfer_feature_layer is None:
        output = base.layers[-2].output
    else:
        # saved models carry suffixed names ('flatten_5'), so match on prefix
        layer = next(l for l in base.layers if l.name.startswith(transfer_feature_layer))
        output = layer.output
    trunk = models.Model(inputs=base.inputs, outputs=output, name='trunk')
    trunk.trainable = False  # must happen before compile or the freeze doesn't take
    return trunk


def attach_head(trunk, categories_count):
    """Both models share this head -- if it differs, the comparison proves nothing.

    The trunk is invoked with training=False. trainable=False freezes the *weights* but does
    NOT put Dropout into inference mode, so without this the frozen trunk keeps dropping 25% of
    its conv activations and 50% of its 64-unit output on every training step -- measured at
    82% zeros during fit versus 64% at inference. That destroys most of the transferred signal
    and flattens the learning curve to near chance.
    """
    inputs = Input(shape=trunk.input_shape[1:])
    features = trunk(inputs, training=False)
    hidden = layers.Dense(64, activation='relu')(features)
    outputs = layers.Dense(categories_count, activation='softmax')(hidden)
    return models.Model(inputs, outputs)


def compile_head(model):
    model.compile(
        optimizer=RMSprop(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )


class TransferedModel(Model):
    def _define_model(self, input_shape, categories_count):
        self.model = attach_head(load_trunk(), categories_count)

    def _compile_model(self):
        compile_head(self.model)

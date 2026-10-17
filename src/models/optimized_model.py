from models.model import Model
from tensorflow.keras import Sequential, layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.optimizers import Adam


class OptimizedModel(Model):
    """Section 6 result -- hp_search.py config C3, 0.7569 test accuracy at 124,195 parameters.

    Three changes from BasicModel, one per search stage:
      stage A  Adam instead of RMSprop at the same 1e-3 learning rate (+3.3 points)
      stage B  wider first conv block, 8/16/32/32 -> 16/32/32/32 (neutral, within noise)
      stage C  two dropout layers, 0.25 after the conv stack and 0.5 before the output (+1.6)

    Five-block variants were tested in stage B and were consistently worse -- the extra pooling
    stage cuts the feature map to 3x3 and discards detail these three expressions depend on.
    """

    def _define_model(self, input_shape, categories_count):
        self.model = Sequential([
            Rescaling(1. / 255, input_shape=input_shape),

            layers.Conv2D(16, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),

            layers.Dropout(0.25),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(categories_count, activation='softmax'),
        ])

    def _compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy'],
        )

"""Section 8 wiring check -- `python test_transfer.py`.

Uses fake 2-class data and an untrained stand-in base model, so it runs without the real
datasets. Catches the failures that are silent otherwise: the trunk not freezing (freeze set
after compile), images not resized to the trunk's input shape, integer instead of categorical
labels, and random_model not actually randomizing.
"""
import os
import shutil
import tempfile

import numpy as np
import tensorflow as tf

import config

tmp = os.path.join(tempfile.gettempdir(), 'p6_transfer_check')
shutil.rmtree(tmp, ignore_errors=True)
for split, n in (('train', 40), ('test', 10)):
    for cls in config.transfer_categories:
        d = os.path.join(tmp, split, cls)
        os.makedirs(d)
        for i in range(n):
            img = tf.cast(tf.random.uniform((64, 64, 3), 0, 255), tf.uint8)
            tf.io.write_file(os.path.join(d, '%d.jpg' % i), tf.io.encode_jpeg(img))

config.transfer_train_directory = os.path.join(tmp, 'train')
config.transfer_test_directory = os.path.join(tmp, 'test')
config.transfer_base_model = os.path.join(tmp, 'base.keras')

from models.basic_model import BasicModel
BasicModel((150, 150, 3), 3).save_model(config.transfer_base_model)

import preprocess
preprocess.transfer_train_directory = config.transfer_train_directory
preprocess.transfer_test_directory = config.transfer_test_directory
train_dataset, validation_dataset, test_dataset = preprocess.get_transfer_datasets()

x, y = next(iter(train_dataset))
assert x.shape[1:] == (150, 150, 3), 'transfer images not resized to the trunk input shape'
assert y.shape[1] == len(config.transfer_categories), 'labels are not categorical'

from models.transfered_model import TransferedModel
from models.random_model import RandomModel

trunk = lambda m: next(l for l in m.layers if l.name == 'trunk')

built = {}
for name, model_class in (('transfered_model', TransferedModel), ('random_model', RandomModel)):
    model = model_class((150, 150, 3), 2)
    trainable = int(sum(np.prod(w.shape) for w in model.model.trainable_weights))
    frozen = int(sum(np.prod(w.shape) for w in model.model.non_trainable_weights))
    # compare against the trunk itself, not a size threshold: with transfer_feature_layer =
    # 'flatten' the 100k-parameter dense layer moves into the head, so the head outweighs the
    # trunk even though the freeze took
    assert frozen == trunk(model.model).count_params(), '%s: the trunk did not freeze' % name
    model.train_model(train_dataset, validation_dataset, epochs=1)
    built[name] = model.model
    print('%s: trainable %s | frozen %s' % (name, format(trainable, ','), format(frozen, ',')))

a, b = built['transfered_model'], built['random_model']
shape = lambda m: [(type(l).__name__, tuple(l.output_shape[1:])) for l in m.layers]
assert shape(a) == shape(b), 'the two models must differ only in trunk weights'
assert a.count_params() == b.count_params(), 'parameter counts diverged'

assert not np.allclose(trunk(a).get_weights()[0], trunk(b).get_weights()[0]), \
    'random_model did not randomize the trunk'

# the frozen trunk must be in inference mode during training, or its dropout layers keep
# firing and shred the transferred features
x = tf.random.uniform((2, 150, 150, 3), 0, 255)
assert np.allclose(a(x, training=True).numpy(), a(x, training=True).numpy()), \
    'trunk is stochastic under training=True -- its dropout is still active'

shutil.rmtree(tmp, ignore_errors=True)
print('\nOK: identical architecture and head, frozen trunk, different trunk weights')

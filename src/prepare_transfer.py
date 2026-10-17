"""Build the section 8 transfer dataset -- `python src/prepare_transfer.py [PetImages dir]`.

Kaggle's Dogs vs Cats is the Microsoft Asirra set, distributed as PetImages/{Cat,Dog}. A few
thousand of those files are truncated or not actually JPEGs and will kill training partway
through an epoch, so every image is decoded before it is accepted.

Produces the layout config.py expects:
    data/transfer/train/{cat,dog}   TRAIN_PER_CLASS each
    data/transfer/test/{cat,dog}    TEST_PER_CLASS each
"""
import os
import shutil
import sys

import tensorflow as tf

from config import (transfer_train_directory, transfer_test_directory, transfer_categories,
                    data_directory)

TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 200


def usable(path):
    try:
        img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        return img.shape[0] > 0 and img.shape[1] > 0
    except Exception:
        return False


def build(source_directory):
    needed = TRAIN_PER_CLASS + TEST_PER_CLASS
    for category in transfer_categories:
        source_category_dir = os.path.join(source_directory, category.capitalize())
        files = sorted(os.listdir(source_category_dir))

        kept, skipped = [], 0
        for name in files:
            if len(kept) == needed:
                break
            path = os.path.join(source_category_dir, name)
            if usable(path):
                kept.append(path)
            else:
                skipped += 1

        assert len(kept) == needed, \
            'only found {} usable {} images, needed {}'.format(len(kept), category, needed)

        for directory, paths in ((transfer_train_directory, kept[:TRAIN_PER_CLASS]),
                                 (transfer_test_directory, kept[TRAIN_PER_CLASS:])):
            target = os.path.join(directory, category)
            os.makedirs(target, exist_ok=True)
            for path in paths:
                shutil.copy(path, os.path.join(target, os.path.basename(path)))

        print('{}: {} train, {} test (skipped {} unreadable)'.format(
            category, TRAIN_PER_CLASS, TEST_PER_CLASS, skipped))


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(data_directory, 'PetImages'))

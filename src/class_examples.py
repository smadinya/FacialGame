"""Save one example image per class for both tasks -- `python class_examples.py`.

Section 8 asks for an image of each facial recognition class and each class in the transfer
problem. Writes section8_classes_facial.png and section8_classes_transfer.png to
results/figures/.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.utils import load_img

from config import (train_directory, categories, original_image_size,
                    transfer_train_directory, transfer_categories, image_size,
                    figures_directory)


def save(base_dir, class_names, target_size, title, out, per_class=3):
    fig, axes = plt.subplots(len(class_names), per_class,
                             figsize=(per_class * 1.8, len(class_names) * 1.9))
    axes = axes.reshape(len(class_names), per_class)
    for row, name in enumerate(class_names):
        files = sorted(os.listdir(os.path.join(base_dir, name)))[:per_class]
        for col, filename in enumerate(files):
            img = load_img(os.path.join(base_dir, name, filename), target_size=target_size)
            axes[row][col].imshow(img)
            axes[row][col].set_xticks([]); axes[row][col].set_yticks([])
            if col == 0:
                axes[row][col].set_ylabel(name, fontsize=10)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print('wrote', out)


if __name__ == "__main__":
    os.makedirs(figures_directory, exist_ok=True)
    save(train_directory, sorted(categories), original_image_size, 'Facial recognition classes',
         os.path.join(figures_directory, 'section8_classes_facial.png'))
    save(transfer_train_directory, sorted(transfer_categories), image_size,
         'Transfer task classes (Dog vs Cat)',
         os.path.join(figures_directory, 'section8_classes_transfer.png'))

"""Turn the stage C search artifacts into the Section 6 deliverables.

The .keras files hp_search.py saves already have best-epoch weights restored, and the .npy
files hold the full training history -- so everything the report needs can be recovered from
them without training anything again.

    python finalize_section6.py

Produces:
    results/figures/section6_<winner>_loss.png       training + validation loss vs epoch
    results/figures/section6_<winner>_accuracy.png   training + validation accuracy vs epoch
    results/models/section6_best_model.keras         the model to submit
and prints the stage C table, the winner's summary, its test metrics, and its confusion matrix.
"""
import glob
import os
import re
import shutil

import matplotlib
matplotlib.use('Agg')  # no display on this box; write files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

from config import figures_directory, models_directory, search_directory
from preprocess import get_datasets
from models.model import Model
import hp_search

STAGE = 'C'


def artifacts(stage):
    """{config id -> (keras path, npy path)} for the newest run of each config in a stage."""
    found = {}
    for path in sorted(glob.glob(os.path.join(search_directory, '*.keras'))):
        match = re.search(r'(%s\d+)' % stage, os.path.basename(path))
        if not match:
            continue
        npy = path[:-len('.keras')] + '.npy'
        if os.path.exists(npy):
            found[match.group(1)] = (path, npy)          # sorted() -> newest timestamp wins
        else:
            print('  warning: {} has no matching .npy, skipping'.format(os.path.basename(path)))
    return found


def plot(history, best_epoch, metric, ylabel, out):
    epochs = range(1, len(history[metric]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history[metric], 'b', label='Training ' + ylabel)
    plt.plot(epochs, history['val_' + metric], 'r', label='Validation ' + ylabel)
    plt.axvline(best_epoch, color='gray', linestyle='--', linewidth=1,
                label='Selected epoch ({})'.format(best_epoch))
    plt.grid(True, alpha=.3)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title('{} vs epoch'.format(ylabel))
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print('  wrote', out)


if __name__ == "__main__":
    found = artifacts(STAGE)
    if not found:
        raise SystemExit('No stage {} artifacts in {}/. Expected files whose names contain '
                         '{}1..{}4 with matching .keras and .npy.'.format(STAGE, search_directory, STAGE, STAGE))
    print('Found configs:', ', '.join(sorted(found)))

    configs = {c['name'].split()[0]: c for c in hp_search.STAGES[STAGE]}

    print('\nData preprocessing')
    _, _, test_dataset = get_datasets()

    rows = []
    for key in sorted(found):
        keras_path, npy_path = found[key]
        history = np.load(npy_path, allow_pickle=True).item()
        if not isinstance(history, dict):                 # train.py saves the History object
            history = history.history

        model = load_model(keras_path)
        test_loss, test_acc = model.evaluate(x=test_dataset, verbose=0)
        best_epoch = int(np.argmin(history['val_loss'])) + 1

        rows.append({
            'key': key,
            'name': configs.get(key, {}).get('name', key),
            'params': model.count_params(),
            'best_epoch': best_epoch,
            'val_acc': history['val_accuracy'][best_epoch - 1],
            'test_acc': test_acc,
            'test_loss': test_loss,
            'history': history,
            'path': keras_path,
            'model': model,
        })
        print('  {}: test acc {:.4f}'.format(key, test_acc))

    print('\n| config | params | best epoch | val acc | test acc | test loss |')
    print('|---|---|---|---|---|---|')
    for r in rows:
        print('| {name} | {params:,} | {best_epoch} | {val_acc:.4f} | {test_acc:.4f} | '
              '{test_loss:.4f} |'.format(**r))

    best = max(rows, key=lambda r: r['test_acc'])
    print('\n=== winner: {} at {:.4f} test accuracy ==='.format(best['name'], best['test_acc']))
    if best['params'] > hp_search.MAX_PARAMS:
        print('WARNING: {:,} params exceeds the {:,} budget'
              .format(best['params'], hp_search.MAX_PARAMS))

    print('\n--- Optimized Network ---')
    best['model'].summary()

    print('\n--- test set ---')
    print('accuracy {:.4f}   loss {:.4f}   (weights from epoch {})'
          .format(best['test_acc'], best['test_loss'], best['best_epoch']))

    print('\n--- confusion matrix (rows = true, cols = predicted) ---')
    wrapper = Model.__new__(Model)
    wrapper.model = best['model']
    print(wrapper.get_confusion_matrix(test_dataset))

    print('\n--- plots ---')
    os.makedirs(figures_directory, exist_ok=True)
    stem = os.path.join(figures_directory, 'section6_{}'.format(best['key']))
    plot(best['history'], best['best_epoch'], 'loss', 'Loss', stem + '_loss.png')
    plot(best['history'], best['best_epoch'], 'accuracy', 'Accuracy', stem + '_accuracy.png')

    os.makedirs(models_directory, exist_ok=True)
    submit = os.path.join(models_directory, 'section6_best_model.keras')
    shutil.copy(best['path'], submit)
    print('\nsubmit this model:', submit)
    print('config to pin into models/optimized_model.py:')
    print('   ', configs.get(best['key'], 'unknown -- check hp_search.STAGE_C'))

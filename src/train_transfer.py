from preprocess import get_transfer_datasets
from models.transfered_model import TransferedModel
from models.random_model import RandomModel
from config import image_size, categories, figures_directory, models_directory
from config import transfer_feature_layer
import matplotlib.pyplot as plt
import os
import time

# Your code should change these values based on your choice of dataset for the transfer task
# -------------
input_shape = (image_size[0], image_size[1], 3)
categories_count = 2
transfer_task = 'Dog vs Cat'
# -------------

models = {
    'transfered_model': TransferedModel,
    'random_model': RandomModel,
}

# note the argument order: models below iterates transfered_model first, so histories[0] is
# the transfer run. as shipped this function labelled its first argument 'Initial', which put
# the legend on the wrong curves.
def plot_history_diff(transfered_hist, initial_hist):
    val_acc_transfered = transfered_hist.history['val_accuracy']
    val_acc_initial = initial_hist.history['val_accuracy']

    epochs_transfered = range(1, len(val_acc_transfered) + 1)
    epochs_initial = range(1, len(val_acc_initial) + 1)
    assert epochs_initial == epochs_transfered, "The two models have been tried with different epochs"

    plt.figure(figsize = (9, 5.5))   # 24x6 is unreadable once embedded in the report
    plt.plot(epochs_transfered, val_acc_transfered, 'r', label = 'With Transfer')
    plt.plot(epochs_initial, val_acc_initial, 'b', label = 'Without Transfer (random frozen trunk)')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Far Transfer from Facial Recognition to {}'.format(transfer_task))
    plt.tight_layout()
    # suffixed by the transferred layer so the bottleneck and flatten runs don't overwrite
    # each other -- the report shows both
    os.makedirs(figures_directory, exist_ok=True)
    out = os.path.join(figures_directory, 'section8_far_transfer_{}.png'.format(
        transfer_feature_layer or 'bottleneck'))
    plt.savefig(out, dpi=150)   # save before show(); show() is a no-op with no display
    print('* Wrote {}'.format(out))
    plt.show()

if __name__ == "__main__":
    # Your code should change the number of epochs
    epochs = 20  # frozen trunk + tiny head, so epochs are cheap
    print('* Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_transfer_datasets()
    histories = []
    test_results = {}
    for name, model_class in models.items():
        print('* Training {} for {} epochs'.format(name, epochs))
        model = model_class(input_shape, categories_count)
        model.print_summary()
        history = model.train_model(train_dataset, validation_dataset, epochs)
        histories.append(history)
        print('* Evaluating {}'.format(name))
        test_results[name] = model.evaluate(test_dataset)
        print('* Confusion Matrix for {}'.format(name))
        print(model.get_confusion_matrix(test_dataset))
        os.makedirs(models_directory, exist_ok=True)
        saved = os.path.join(models_directory, 'section8_{}.keras'.format(name))
        model.save_model(saved)
        print('* Saved {}'.format(saved))
    assert len(histories) == 2, "The number of trained models is not equal to two"

    print('\n* Test set accuracy on {}'.format(transfer_task))
    for name, (loss, acc) in test_results.items():
        label = 'with transfer' if name == 'transfered_model' else 'without transfer'
        print('  {:<16} ({:<17}): accuracy {:.4f}, loss {:.4f}'.format(name, label, acc, loss))

    plot_history_diff(*histories)

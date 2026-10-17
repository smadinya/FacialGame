import numpy as np
from preprocess import get_datasets
from models.basic_model import BasicModel
from models.optimized_model import OptimizedModel
from models.model import Model
from config import image_size, models_directory
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import time
import os

input_shape = (image_size[0], image_size[1], 3)
categories_count = 3

models = {
    'basic_model': BasicModel,        # section 4/5
    'optimized_model': OptimizedModel, # section 6
}

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize = (24, 6))
    plt.subplot(1,2,1)
    plt.plot(epochs, acc, 'b', label = 'Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label = 'Validation Accuracy')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')

    plt.subplot(1,2,2)
    plt.plot(epochs, loss, 'b', label = 'Training Loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation Loss')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()

if __name__ == "__main__":
    # if you want to load your model later, you can use:
    # model = Model.load_model("name_of_your_model.keras")
    # to load your history and plot it again, you can use:
    # history = np.load('results/models/name_of_your_model.npy',allow_pickle='TRUE').item()
    # plot_history(history)
    epochs = 40
    print('Data preprocessing')
    train_dataset, validation_dataset, test_dataset = get_datasets()
    name = 'optimized_model'  # 'basic_model' for the section 4/5 run
    model_class = models[name]
    print('Training {} for up to {} epochs'.format(name, epochs))
    model = model_class(input_shape, categories_count)
    model.print_summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1,
    )

    history = model.train_model(train_dataset, validation_dataset, epochs, callbacks=[early_stopping])
    print('Evaluating {}'.format(name))
    model.evaluate(test_dataset)
    print('Confusion Matrix for {}'.format(name))
    print(model.get_confusion_matrix(test_dataset))

    os.makedirs(models_directory, exist_ok=True)
    model_name = '{}_{}_epochs_timestamp_{}'.format(name, epochs, int(time.time()))
    filename = os.path.join(models_directory, '{}.keras'.format(model_name))
    model.save_model(filename)
    np.save(os.path.join(models_directory, '{}.npy'.format(model_name)), history)
    print('Model saved as {}'.format(filename))
    print('To play the game with it, point game_model in config.py at this file.')

    plot_history(history)

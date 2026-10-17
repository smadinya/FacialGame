from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split
from config import categories, transfer_categories
from config import transfer_train_directory, transfer_test_directory, transfer_batch_size

def _split_data(train_directory, test_directory, batch_size, validation_split, class_names):
    # class_names is passed sorted, which is the order image_dataset_from_directory would pick
    # on its own -- so label indices are unchanged, but a stray subdirectory (a leftover
    # category, .ipynb_checkpoints) now fails here with a clear message naming it, instead of
    # silently adding a class and blowing up as a logits/labels shape mismatch mid-training.
    class_names = sorted(class_names)
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47,
        class_names=class_names
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        class_names=class_names
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split, categories)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    # same image_size and rgb color_mode as the facial task -- the frozen trunk's input shape
    # is fixed, so the transfer images have to be resized to match it
    train_dataset, validation_dataset, test_dataset = \
        _split_data(transfer_train_directory, transfer_test_directory,
                    transfer_batch_size, validation_split, transfer_categories)
    return train_dataset, validation_dataset, test_dataset
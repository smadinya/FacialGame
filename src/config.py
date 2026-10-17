import os

# Every path is anchored to the repository root, so scripts can be run from any directory.
root_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_directory = os.path.join(root_directory, 'data')
results_directory = os.path.join(root_directory, 'results')
figures_directory = os.path.join(results_directory, 'figures')
models_directory = os.path.join(results_directory, 'models')
search_directory = os.path.join(results_directory, 'search')

# Alphabetical, because that is the label order image_dataset_from_directory assigns: the trained
# models output 0 = happy, 1 = neutral, 2 = surprise. The game reads this list to tell the player
# which expression selects which row/column, so it must match the model's output order.
categories = ['happy', 'neutral', 'surprise']

train_directory = os.path.join(data_directory, 'train')
test_directory = os.path.join(data_directory, 'test')

train_size = 5000
original_image_size = (48, 48)
image_size = (150, 150)
batch_size = 128
validation_split = 0.2

# section 8 transfer task
transfer_train_directory = os.path.join(data_directory, 'transfer', 'train')
transfer_test_directory = os.path.join(data_directory, 'transfer', 'test')
transfer_categories = ['cat', 'dog']
transfer_batch_size = 32  # ~2000 images, not 5000 -- 128 leaves too few steps per epoch

# the section 6 model both transfer models are built from. pinned to an explicit results/
# path so that retraining never silently changes what section 8 transferred from.
transfer_base_model = os.path.join(models_directory, 'section6_best_model.keras')

# Which layer of the base model to transfer features from.
#   None      -- everything except the final softmax, i.e. the 64-unit bottleneck. This is the
#                literal reading of the writeup ("eliminate the final softmax layer").
#   'flatten' -- the convolutional feature map instead, 1568 units, dropping the task-specific
#                fully connected layer as well.
# The bottleneck is squeezed to exactly what separates three facial expressions, so it carries
# very little that generalises; the conv features are far more generic. See the report.
transfer_feature_layer = 'flatten'

# the model the webcam player loads to read expressions during the game
game_model = os.path.join(models_directory, 'section6_best_model.keras')

BOARD_SIZE = 3

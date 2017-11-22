import sys, os

sys.path.append(os.path.realpath('..'))

import data_utils

dataset, image_size, num_of_classes, num_of_channels = data_utils.prepare_not_mnist_dataset(root_dir="..")
dataset = data_utils.reformat(dataset, image_size, num_of_channels, num_of_classes)

print("After reformat:")
print('Training set', dataset.train_dataset.shape, dataset.train_labels.shape)
print('Validation set', dataset.valid_dataset.shape, dataset.valid_labels.shape)
print('Test set', dataset.test_dataset.shape, dataset.test_labels.shape)
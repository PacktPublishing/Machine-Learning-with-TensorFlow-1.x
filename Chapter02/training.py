import sys, os
import tensorflow as tf
import numpy as np

sys.path.append(os.path.realpath('..'))

import data_utils
import logmanager

batch_size = 128
num_steps = 10000
learning_rate = 0.3
data_showing_step = 500


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def nn_model(data, weights, biases):
    layer_fc1 = tf.matmul(data, weights['fc1']) + biases['fc1']
    relu_layer = tf.nn.relu(layer_fc1)
    return tf.matmul(relu_layer, weights['fc2']) + biases['fc2']


dataset, image_size, num_of_classes, num_of_channels = data_utils.prepare_not_mnist_dataset(root_dir="..")
dataset = data_utils.reformat(dataset, image_size, num_of_channels, num_of_classes)

print('Training set', dataset.train_dataset.shape, dataset.train_labels.shape)
print('Validation set', dataset.valid_dataset.shape, dataset.valid_labels.shape)
print('Test set', dataset.test_dataset.shape, dataset.test_labels.shape)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size * num_of_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes))
    tf_valid_dataset = tf.constant(dataset.valid_dataset)
    tf_test_dataset = tf.constant(dataset.test_dataset)

    # Variables.
    weights = {
        'fc1': tf.Variable(tf.truncated_normal([image_size * image_size * num_of_channels, num_of_classes])),
        'fc2': tf.Variable(tf.truncated_normal([num_of_classes, num_of_classes]))
    }
    biases = {
        'fc1': tf.Variable(tf.zeros([num_of_classes])),
        'fc2': tf.Variable(tf.zeros([num_of_classes]))
    }

    # Training computation.
    logits = nn_model(tf_train_dataset, weights, biases)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases))
    test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases))

with tf.Session(graph=graph) as session:
    session.run(tf.global_variables_initializer())
    print("Initialized")
    for step in range(num_steps + 1):
        sys.stdout.write('Training on batch %d of %d\r' % (step + 1, num_steps))
        sys.stdout.flush()
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (dataset.train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = dataset.train_dataset[offset:(offset + batch_size), :]
        batch_labels = dataset.train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % data_showing_step == 0:
            acc_minibatch = accuracy(predictions, batch_labels)
            acc_val = accuracy(valid_prediction.eval(), dataset.valid_labels)
            logmanager.logger.info('# %03d  Acc Train: %03.2f%%  Acc Val: %03.2f%% Loss %f' % (
                step, acc_minibatch, acc_val, l))
    logmanager.logger.info("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), dataset.test_labels))

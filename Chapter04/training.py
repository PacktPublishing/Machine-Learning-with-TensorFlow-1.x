import sys, os
import tensorflow as tf
import numpy as np

sys.path.append(os.path.realpath('..'))

import data_utils
import logmanager

import math

num_steps = 30000
learning_rate = 0.1
data_showing_step = 500
model_saving_step = 2000
log_location = '/tmp/nn_log'

SEED = 11215

batch_size = 32
patch_size = 5
depth_inc = 4
num_hidden_inc = 32
dropout_prob = 0.8
conv_layers = 3
stddev = 0.1


# For same padding the output width or height = ceil(width or height / stride) respectively
def fc_first_layer_dimen(image_size, layers):
    output = image_size
    for x in range(layers):
        output = math.ceil(output/2.0)
    return int(output)


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


def nn_model(data, weights, biases, TRAIN=False):
    with tf.name_scope('Layer_1') as scope:
        conv = tf.nn.conv2d(data, weights['conv1'], strides=[1, 1, 1, 1], padding='SAME', name='conv1')
        bias_add = tf.nn.bias_add(conv, biases['conv1'], name='bias_add_1')
        relu = tf.nn.relu(bias_add, name='relu_1')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    with tf.name_scope('Layer_2') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME', name='conv2')
        bias_add = tf.nn.bias_add(conv, biases['conv2'], name='bias_add_2')
        relu = tf.nn.relu(bias_add, name='relu_2')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name=scope)

    with tf.name_scope('Layer_3') as scope:
        conv = tf.nn.conv2d(max_pool, weights['conv3'], strides=[1, 1, 1, 1], padding='SAME', name='conv3')
        bias_add = tf.nn.bias_add(conv, biases['conv3'], name='bias_add_3')
        relu = tf.nn.relu(bias_add, name='relu_3')
        max_pool = tf.nn.max_pool(relu, ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1], padding='SAME', name=scope)
        if TRAIN:
            max_pool = tf.nn.dropout(max_pool, dropout_prob, seed=SEED, name="dropout_3")

    print("max_pool ", max_pool.get_shape())
    shape = max_pool.get_shape().as_list()
    reshape = tf.reshape(max_pool, [shape[0], shape[1] * shape[2] * shape[3]])

    with tf.name_scope('FC_Layer_1') as scope:
        matmul = tf.matmul(reshape, weights['fc1'], name='fc1_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc1'], name='fc1_bias_add')
        relu = tf.nn.relu(bias_add, name=scope)

    with tf.name_scope('FC_Layer_2') as scope:
        matmul = tf.matmul(relu, weights['fc2'], name='fc2_matmul')
        bias_add = tf.nn.bias_add(matmul, biases['fc2'], name='fc2_bias_add')
        layer_fc2 = tf.nn.relu(bias_add, name=scope)

    return layer_fc2

dataset, image_size, num_of_classes, num_channels = data_utils.prepare_not_mnist_dataset(root_dir="..")
dataset = data_utils.reformat(dataset, image_size, num_channels, num_of_classes, flatten=False)

print('Training set', dataset.train_dataset.shape, dataset.train_labels.shape)
print('Validation set', dataset.valid_dataset.shape, dataset.valid_labels.shape)
print('Test set', dataset.test_dataset.shape, dataset.test_labels.shape)

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size, image_size, num_channels), name="TRAIN_DATASET")
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_of_classes), name="TRAIN_LABEL")
    tf_valid_dataset = tf.constant(dataset.valid_dataset, name='VALID_DATASET')
    tf_test_dataset = tf.constant(dataset.test_dataset, name='TEST_DATASET')

    print ("Image Size", image_size)
    print ("Conv Layers", conv_layers)
    print ("fc_first_layer_dimen", fc_first_layer_dimen(image_size, conv_layers))

    # Variables.
    weights = {
        'conv1': tf.Variable(tf.truncated_normal(shape=[patch_size, patch_size, num_channels, depth_inc], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights_conv1'),
        'conv2': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_inc, depth_inc], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights_conv2'),
        'conv3': tf.Variable(tf.truncated_normal([patch_size, patch_size, depth_inc, depth_inc], dtype=tf.float32,
                                                 stddev=stddev, seed=SEED), name='weights_conv3'),
        'fc1': tf.Variable(tf.truncated_normal([(fc_first_layer_dimen(image_size, conv_layers) ** 2) * depth_inc,
                                                num_hidden_inc], dtype=tf.float32,
                                               stddev=stddev, seed=SEED), name='weights_fc1'),
        'fc2': tf.Variable(tf.truncated_normal([num_hidden_inc, num_of_classes], dtype=tf.float32,
                                               stddev=stddev, seed=SEED), name='weights_fc2')
    }
    biases = {
        'conv1': tf.Variable(tf.zeros(shape=[depth_inc], dtype=tf.float32), name='biases_conv1'),
        'conv2': tf.Variable(tf.zeros(shape=[depth_inc], dtype=tf.float32), name='biases_conv2'),
        'conv3': tf.Variable(tf.zeros(shape=[depth_inc], dtype=tf.float32), name='biases_conv3'),
        'fc1': tf.Variable(tf.zeros(shape=[num_hidden_inc], dtype=tf.float32), name='biases_fc1'),
        'fc2': tf.Variable(tf.zeros(shape=[num_of_classes], dtype=tf.float32), name='biases_fc2'),
    }

    # Training computation.
    logits = nn_model(tf_train_dataset, weights, biases, True)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    regularizers = (tf.nn.l2_loss(weights['fc1']) +
                    tf.nn.l2_loss(biases['fc1']) +
                    tf.nn.l2_loss(weights['fc2']) + tf.nn.l2_loss(biases['fc2']))
    # Add the regularization term to the loss.
    loss += 5e-4 * regularizers
    tf.summary.scalar("loss", loss)

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(nn_model(tf_train_dataset, weights, biases, TRAIN=False))
    valid_prediction = tf.nn.softmax(nn_model(tf_valid_dataset, weights, biases))
    test_prediction = tf.nn.softmax(nn_model(tf_test_dataset, weights, biases))

with tf.Session(graph=graph) as session:
    writer = tf.summary.FileWriter(log_location, session.graph)
    saver = tf.train.Saver(max_to_keep=5)

    merged = tf.summary.merge_all()
    tf.global_variables_initializer().run()
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
        summary_result, _, l, predictions = session.run([merged, optimizer, loss, train_prediction], feed_dict=feed_dict)
        writer.add_summary(summary_result, step)

        if step % data_showing_step == 0:
            acc_minibatch = accuracy(predictions, batch_labels)
            acc_val = accuracy(valid_prediction.eval(), dataset.valid_labels)
            logmanager.logger.info('# %03d  Acc Train: %03.2f%%  Acc Val: %03.2f%% Loss %f' % (
                step, acc_minibatch, acc_val, l))

        if step % model_saving_step == 0 or step == num_steps + 1:
            path = saver.save(session, os.path.join(log_location, "model.ckpt"), global_step=step)
            logmanager.logger.info('Model saved in file: %s' % path)

    logmanager.logger.info("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), dataset.test_labels))

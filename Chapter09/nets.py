import tensorflow as tf
import numpy as np
import os
from utils import debug_print
from scipy.misc import imread, imresize


NUM_CLASSES = 37


def _conv2d(input_data, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding="SAME"):
    c_i = input_data.get_shape()[-1].value
    convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        tf.Variable
        weights = tf.get_variable(name="kernel", shape=[k_h, k_w, c_i, c_o],
                                  initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32))
        conv = convolve(input_data, weights)
        biases = tf.get_variable(name="bias", shape=[c_o], dtype=tf.float32,
                                 initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.bias_add(conv, biases)
        if relu:
            output = tf.nn.relu(output, name=scope.name)
        return output


def _max_pool(input_data, k_h, k_w, s_h, s_w, name, padding="SAME"):
    return tf.nn.max_pool(input_data, ksize=[1, k_h, k_w, 1],
                          strides=[1, s_h, s_w, 1], padding=padding, name=name)


def _fully_connected(input_data, num_output, name, relu=True):
    with tf.variable_scope(name) as scope:
        input_shape = input_data.get_shape()
        if input_shape.ndims == 4:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(input_data, [-1, dim])
        else:
            feed_in, dim = (input_data, input_shape[-1].value)
        weights = tf.get_variable(name="kernel", shape=[dim, num_output],
                                  initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32))
        biases = tf.get_variable(name="bias", shape=[num_output], dtype=tf.float32,
                                 initializer=tf.constant_initializer(value=0.0))
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        output = op(feed_in, weights, biases, name=scope.name)
        return output


def _softmax(input_data, name):
    return tf.nn.softmax(input_data, name=name)


def inference(images, is_training=False):
    with tf.name_scope("preprocess"):
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        input_images = images - mean
    conv1_1 = _conv2d(input_images, 3, 3, 64, 1, 1, name="conv1_1")
    conv1_2 = _conv2d(conv1_1, 3, 3, 64, 1, 1, name="conv1_2")
    pool1 = _max_pool(conv1_2, 2, 2, 2, 2, name="pool1")

    conv2_1 = _conv2d(pool1, 3, 3, 128, 1, 1, name="conv2_1")
    conv2_2 = _conv2d(conv2_1, 3, 3, 128, 1, 1, name="conv2_2")
    pool2 = _max_pool(conv2_2, 2, 2, 2, 2, name="pool2")

    conv3_1 = _conv2d(pool2, 3, 3, 256, 1, 1, name="conv3_1")
    conv3_2 = _conv2d(conv3_1, 3, 3, 256, 1, 1, name="conv3_2")
    conv3_3 = _conv2d(conv3_2, 3, 3, 256, 1, 1, name="conv3_3")
    pool3 = _max_pool(conv3_3, 2, 2, 2, 2, name="pool3")

    conv4_1 = _conv2d(pool3, 3, 3, 512, 1, 1, name="conv4_1")
    conv4_2 = _conv2d(conv4_1, 3, 3, 512, 1, 1, name="conv4_2")
    conv4_3 = _conv2d(conv4_2, 3, 3, 512, 1, 1, name="conv4_3")
    pool4 = _max_pool(conv4_3, 2, 2, 2, 2, name="pool4")

    conv5_1 = _conv2d(pool4, 3, 3, 512, 1, 1, name="conv5_1")
    conv5_2 = _conv2d(conv5_1, 3, 3, 512, 1, 1, name="conv5_2")
    conv5_3 = _conv2d(conv5_2, 3, 3, 512, 1, 1, name="conv5_3")
    pool5 = _max_pool(conv5_3, 2, 2, 2, 2, name="pool5")

    fc6 = _fully_connected(pool5, 4096, name="fc6")
    fc7 = _fully_connected(fc6, 4096, name="fc7")
    if is_training:
        fc7 = tf.nn.dropout(fc7, keep_prob=0.5)
    fc8 = _fully_connected(fc7, 1000, name='fc8', relu=False)
    return fc8


def load_caffe_weights(path, sess, ignore_missing=False):
    print("Load caffe weights from ", path)
    data_dict = np.load(path).item()
    for op_name in data_dict:
        with tf.variable_scope(op_name, reuse=True):
            for param_name, data in data_dict[op_name].iteritems():
                try:
                    var = tf.get_variable(param_name)
                    sess.run(var.assign(data))
                except ValueError as e:
                    if not ignore_missing:
                        print(e)
                        raise e

if __name__ == "__main__":
    SAMPLES_FOLDER = "samples_data"
    with open('%s/imagenet-classes.txt' % SAMPLES_FOLDER, 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())

    inputs = tf.placeholder(tf.float32, [None, 224, 224, 3], name="inputs")
    outputs = inference(inputs)

    debug_print.print_variables(tf.global_variables())
    debug_print.print_variables([inputs, outputs])

    with tf.Session() as sess:
        load_caffe_weights("data/VGG16.npz", sess, ignore_missing=False)

        files = os.listdir(SAMPLES_FOLDER)
        for file_name in files:
            if not file_name.endswith(".jpg"):
                continue
            print("=== Predict %s ==== " % file_name)
            img = imread(os.path.join(SAMPLES_FOLDER, file_name), mode="RGB")
            img = imresize(img, (224, 224))

            prob = sess.run(outputs, feed_dict={inputs: [img]})[0]
            preds = (np.argsort(prob)[::-1])[0:5]

            for p in preds:
                print class_labels[p], prob[p]


    # path = "data/VGG16.npz"
    # data_dict = np.load(path).item()
    # for op_name in data_dict:
    #     print(op_name)
    #     for param_name, data in data_dict[op_name].iteritems():
    #         print("\t" + param_name + "\t" + str(data.shape))

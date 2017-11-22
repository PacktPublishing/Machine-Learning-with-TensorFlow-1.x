import tensorflow as tf
from utils import print_variables, print_layers
from tensorflow.contrib.layers.python.layers.layers import batch_norm


def _conv3d(input_data, k_d, k_h, k_w, c_o, s_d, s_h, s_w, name, relu=True, padding="SAME"):
    c_i = input_data.get_shape()[-1].value
    convolve = lambda i, k: tf.nn.conv3d(i, k, [1, s_d, s_h, s_w, 1], padding=padding)
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name="weights", shape=[k_d, k_h, k_w, c_i, c_o],
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                  initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32))
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        conv = convolve(input_data, weights)
        biases = tf.get_variable(name="biases", shape=[c_o], dtype=tf.float32,
                                 initializer=tf.constant_initializer(value=0.0))
        output = tf.nn.bias_add(conv, biases)
        if relu:
            output = tf.nn.relu(output, name=scope.name)
        return batch_norm(output)


def _max_pool3d(input_data, k_d, k_h, k_w, s_d, s_h, s_w, name, padding="SAME"):
    return tf.nn.max_pool3d(input_data, ksize=[1, k_d, k_h, k_w, 1],
                            strides=[1, s_d, s_h, s_w, 1], padding=padding, name=name)


def _fully_connected(input_data, num_output, name, relu=True):
    with tf.variable_scope(name) as scope:
        input_shape = input_data.get_shape()
        if input_shape.ndims == 5:
            dim = 1
            for d in input_shape[1:].as_list():
                dim *= d
            feed_in = tf.reshape(input_data, [-1, dim])
        else:
            feed_in, dim = (input_data, input_shape[-1].value)
        weights = tf.get_variable(name="weights", shape=[dim, num_output],
                                  regularizer=tf.contrib.layers.l2_regularizer(scale=0.0001),
                                  initializer=tf.truncated_normal_initializer(stddev=1e-1, dtype=tf.float32))
                                  #initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        biases = tf.get_variable(name="biases", shape=[num_output], dtype=tf.float32,
                                 initializer=tf.constant_initializer(value=0.0))
        op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
        output = op(feed_in, weights, biases, name=scope.name)
        return batch_norm(output)


def _softmax(input_data, name):
    return tf.nn.softmax(input_data, name=name)


def inference(input_data, is_training=False):
    conv1 = _conv3d(input_data, 3, 3, 3, 64, 1, 1, 1, "conv1")
    pool1 = _max_pool3d(conv1, 1, 2, 2, 1, 2, 2, "pool1")

    conv2 = _conv3d(pool1, 3, 3, 3, 128, 1, 1, 1, "conv2")
    pool2 = _max_pool3d(conv2, 2, 2, 2, 2, 2, 2, "pool2")
    
    conv3a = _conv3d(pool2, 3, 3, 3, 256, 1, 1, 1, "conv3a")
    conv3b = _conv3d(conv3a, 3, 3, 3, 256, 1, 1, 1, "conv3b")
    pool3 = _max_pool3d(conv3b, 2, 2, 2, 2, 2, 2, "pool3")
    
    conv4a = _conv3d(pool3, 3, 3, 3, 512, 1, 1, 1, "conv4a")
    conv4b = _conv3d(conv4a, 3, 3, 3, 512, 1, 1, 1, "conv4b")
    pool4 = _max_pool3d(conv4b, 2, 2, 2, 2, 2, 2, "pool4")
    
    conv5a = _conv3d(pool4, 3, 3, 3, 512, 1, 1, 1, "conv5a")
    conv5b = _conv3d(conv5a, 3, 3, 3, 512, 1, 1, 1, "conv5b")
    pool5 = _max_pool3d(conv5b, 2, 2, 2, 2, 2, 2, "pool5")

    fc6 = _fully_connected(pool5, 4096, name="fc6")
    fc7 = _fully_connected(fc6, 4096, name="fc7")
    if is_training:
        fc7 = tf.nn.dropout(fc7, keep_prob=0.5)
    fc8 = _fully_connected(fc7, 101, name='fc8', relu=False)
    
    endpoints = dict()
    endpoints["conv1"] = conv1
    endpoints["pool1"] = pool1
    endpoints["conv2"] = conv2
    endpoints["pool2"] = pool2
    endpoints["conv3a"] = conv3a
    endpoints["conv3b"] = conv3b
    endpoints["pool3"] = pool3
    endpoints["conv4a"] = conv4a
    endpoints["conv4b"] = conv4b
    endpoints["pool4"] = pool4
    endpoints["conv5a"] = conv5a
    endpoints["conv5b"] = conv5b
    endpoints["pool5"] = pool5
    endpoints["fc6"] = fc6
    endpoints["fc7"] = fc7
    endpoints["fc8"] = fc8
        
    return fc8, endpoints


if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, [None, 10, 112, 112, 3], name="inputs")
    outputs, endpoints = inference(inputs)

    print_variables(tf.global_variables())
    print_variables([inputs, outputs])
    print_layers(endpoints)

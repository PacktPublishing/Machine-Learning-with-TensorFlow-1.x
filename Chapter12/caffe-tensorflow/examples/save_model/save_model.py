#!/usr/bin/env python

import argparse
import os.path as osp
import sys

import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.training import saver as saver_lib


def save(name, data_input_path):
    def getpardir(path): return osp.split(path)[0]
    sys.path.append(getpardir(getpardir(getpardir(osp.realpath(__file__)))))
    # Import the converted model's class
    caffe_net_module = __import__(name)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        image_input = tf.placeholder(tf.float32, shape=[1, 227, 227, 3], name="data")
        net = caffe_net_module.CaffeNet({'data': image_input})

        # Save protocol buffer
        pb_name = name + '.pb'
        tf.train.write_graph(sess.graph_def, '.', pb_name + 'txt', True)
        tf.train.write_graph(sess.graph_def, '.', pb_name, False)

        if data_input_path is not None:
            # Load the data
            sess.run(tf.global_variables_initializer())
            net.load(data_input_path, sess)
            # Save the data
            saver = saver_lib.Saver(tf.global_variables())
            checkpoint_prefix = osp.join(osp.curdir, name + '.ckpt')
            checkpoint_path = saver.save(sess, checkpoint_prefix)

            # Freeze the graph
            freeze_graph.freeze_graph(pb_name, "",
                                      True, checkpoint_path, 'fc8/fc8',
                                      'save/restore_all', 'save/Const:0',
                                      name + '_frozen.pb', False, "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', help='Name of the converted model')
    parser.add_argument('--data-input-path', help='Converted data input path')
    args = parser.parse_args()
    save(args.name, args.data_input_path)


if __name__ == '__main__':
    main()

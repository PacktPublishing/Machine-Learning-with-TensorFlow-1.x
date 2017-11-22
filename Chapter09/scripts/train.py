import tensorflow as tf
import os
from datetime import datetime
from tqdm import tqdm

import nets, models, datasets

# Dataset
dataset_dir = "data/train_data"
batch_size = 64
image_size = 224

# Learning rate
initial_learning_rate = 0.001
decay_steps = 250
decay_rate = 0.9

# Validation
output_steps = 10  # Number of steps to print output
eval_steps = 20  # Number of steps to perform evaluations

# Training
max_steps = 3000  # Number of steps to perform training
save_steps = 200  # Number of steps to perform saving checkpoints
num_tests = 5  # Number of times to test for test accuracy
max_checkpoints_to_keep = 3
save_dir = "data/checkpoints"
train_vars = 'models/fc8-pets/weights:0,models/fc8-pets/biases:0'

# Export
export_dir = "/tmp/export/"
export_name = "pet-model"
export_version = 2


images, labels = datasets.input_pipeline(dataset_dir, batch_size, is_training=True)
test_images, test_labels = datasets.input_pipeline(dataset_dir, batch_size, is_training=False)

with tf.variable_scope("models") as scope:
    logits = nets.inference(images, is_training=True)
    scope.reuse_variables()
    test_logits = nets.inference(test_images, is_training=False)

total_loss = models.compute_loss(logits, labels)
train_accuracy = models.compute_accuracy(logits, labels)
test_accuracy = models.compute_accuracy(test_logits, test_labels)

global_step = tf.Variable(0, trainable=False)
learning_rate = models.get_learning_rate(global_step, initial_learning_rate, decay_steps, decay_rate)
train_op = models.train(total_loss, learning_rate, global_step, train_vars)

saver = tf.train.Saver(max_to_keep=max_checkpoints_to_keep)
checkpoints_dir = os.path.join(save_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coords = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coords)

    with tf.variable_scope("models"):
       nets.load_caffe_weights("data/VGG16.npz", sess, ignore_missing=True)

    last_saved_test_accuracy = 0
    for i in tqdm(range(max_steps), desc="training"):
        _, loss_value, lr_value = sess.run([train_op, total_loss, learning_rate])

        if (i + 1) % output_steps == 0:
            print("Steps {}: Loss = {:.5f} Learning Rate = {}".format(i + 1, loss_value, lr_value))

        if (i + 1) % eval_steps == 0:
            test_acc, train_acc, loss_value = sess.run([test_accuracy, train_accuracy, total_loss])
            print("Test accuracy {} Train accuracy {} : Loss = {:.5f}".format(test_acc, train_acc, loss_value))

        if (i + 1) % save_steps == 0 or i == max_steps - 1:
            test_acc = 0
            for i in range(num_tests):
                test_acc += sess.run(test_accuracy)
            test_acc /= num_tests
            if test_acc > last_saved_test_accuracy:
                print("Save steps: Test Accuracy {} is higher than {}".format(test_acc, last_saved_test_accuracy))
                last_saved_test_accuracy = test_acc
                saved_file = saver.save(sess,
                                        os.path.join(checkpoints_dir, 'model.ckpt'),
                                        global_step=global_step)
                print("Save steps: Save to file %s " % saved_file)
            else:
                print("Save steps: Test Accuracy {} is not higher than {}".format(test_acc, last_saved_test_accuracy))

    models.export_model(checkpoints_dir, export_dir, export_name, export_version)

    coords.request_stop()
    coords.join(threads)

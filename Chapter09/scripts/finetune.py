import tensorflow as tf
import os
import json
import random
import requests
import shutil
from scipy.misc import imread, imsave
from datetime import datetime
from tqdm import tqdm

import nets, models, datasets


def ensure_folder_exists(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


def download_user_data(url, user_dir, train_ratio=0.8):
    response = requests.get("%s/user-labels" % url)
    data = json.loads(response.text)

    if not os.path.exists(user_dir):
        os.mkdir(user_dir)
    user_dir = ensure_folder_exists(user_dir)
    train_folder = ensure_folder_exists(os.path.join(user_dir, "trainval"))
    test_folder = ensure_folder_exists(os.path.join(user_dir, "test"))

    train_file = open(os.path.join(user_dir, 'trainval.txt'), 'w')
    test_file = open(os.path.join(user_dir, 'test.txt'), 'w')

    for image in data:
        is_train = random.random() < train_ratio
        image_url = image["url"]
        file_name = image_url.split("/")[-1]
        label = image["label"]
        name = image["name"]

        if is_train:
            target_folder = ensure_folder_exists(os.path.join(train_folder, name))
        else:
            target_folder = ensure_folder_exists(os.path.join(test_folder, name))

        target_file = os.path.join(target_folder, file_name) + ".jpg"

        if not os.path.exists(target_file):
            response = requests.get("%s%s" % (url, image_url))
            temp_file_path = "/tmp/%s" % file_name
            with open(temp_file_path, 'wb') as f:
                for chunk in response:
                    f.write(chunk)

            image = imread(temp_file_path)
            imsave(target_file, image)
            os.remove(temp_file_path)
            print("Save file: %s" % target_file)

        label_path = "%s %s\n" % (label, target_file)
        if is_train:
            train_file.write(label_path)
        else:
            test_file.write(label_path)


def make_archive(dir_path):
    return shutil.make_archive(dir_path, 'zip', dir_path)


def archive_and_send_file(source_api, dest_api, ckpt_name, export_dir, export_name, export_version):
    model_dir = os.path.join(export_dir, export_name, str(export_version))
    file_path = make_archive(model_dir)
    print("Zip model: ", file_path)

    data = {
        "link": "{}/{}/{}".format(source_api, export_name, str(export_version) + ".zip"),
        "ckpt_name": ckpt_name,
        "version": export_version,
        "name": export_name,
    }
    r = requests.post(dest_api, data=data)
    print("send_file", r.text)


def get_latest_model(url):
    response = requests.get("%s/model" % url)
    data = json.loads(response.text)
    print(data)
    return data["ckpt_name"], int(data["version"])


# Server info
URL = "http://localhost:5000"
dest_api = URL + "/model"

# Server Endpoints
source_api = "http://1.53.110.161:8181"

# Dataset
dataset_dir = "data/train_data"
user_dir = "data/user_data"
batch_size = 64
image_size = 224

# Learning rate
initial_learning_rate = 0.0001
decay_steps = 250
decay_rate = 0.9

# Validation
output_steps = 10  # Number of steps to print output
eval_steps = 20  # Number of steps to perform evaluations

# Training
max_steps = 2000  # Number of steps to perform training
save_steps = 200  # Number of steps to perform saving checkpoints
num_tests = 5  # Number of times to test for test accuracy
max_checkpoints_to_keep = 3
save_dir = "data/checkpoints"
train_vars = 'models/fc8-pets/weights:0,models/fc8-pets/biases:0'


last_checkpoint_name, last_version = get_latest_model(URL)
last_checkpoint_dir = os.path.join(save_dir, last_checkpoint_name)

# Export
export_dir = "/home/ubuntu/models/"
export_name = "pet-model"
export_version = last_version + 1

# Download user-labels data
download_user_data(URL, user_dir)

images, labels = datasets.input_pipeline(dataset_dir, batch_size, is_training=True, user_dir=user_dir)
test_images, test_labels = datasets.input_pipeline(dataset_dir, batch_size, is_training=False, user_dir=user_dir)

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
checkpoint_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoints_dir = os.path.join(save_dir, checkpoint_name)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
if not os.path.exists(checkpoints_dir):
    os.mkdir(checkpoints_dir)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coords = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coords)

    saver.restore(sess, models.get_model_path_from_ckpt(last_checkpoint_dir))
    sess.run(global_step.assign(0))

    last_saved_test_accuracy = 0
    for i in range(num_tests):
        last_saved_test_accuracy += sess.run(test_accuracy)
    last_saved_test_accuracy /= num_tests
    should_export = False
    print("Last model test accuracy {}".format(last_saved_test_accuracy))
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
                should_export = True
                print("Save steps: Save to file %s " % saved_file)
            else:
                print("Save steps: Test Accuracy {} is not higher than {}".format(test_acc, last_saved_test_accuracy))

    if should_export:
        print("Export model with accuracy ", last_saved_test_accuracy)
        models.export_model(checkpoints_dir, export_dir, export_name, export_version)
        archive_and_send_file(source_api, dest_api, checkpoint_name, export_dir, export_name, export_version)

    coords.request_stop()
    coords.join(threads)


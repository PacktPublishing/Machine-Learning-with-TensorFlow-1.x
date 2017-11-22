import tensorflow as tf
import cv2
import os
import random

from tensorflow.python.ops import data_flow_ops
from utils import lines_from_file


def input_pipeline(input_queue, batch_size=16, num_threads=8, image_size=128, shuffle=True):
    """

    Args:
        input_queue: data queue of image_paths and labels, each image_path contains a list of frame_paths
        num_threads: number of threads for preprocessing
        image_size: size of the expected output
        batch_size: size of the expected batch
        shuffle: bool to shuffle
    Returns:
        frames_batch, labels_batch
    """
    frames_and_labels = []
    for _ in range(num_threads):
        frame_paths, label = input_queue.dequeue()
        frames = []
        for filename in tf.unstack(frame_paths):
            file_contents = tf.read_file(filename)
            image = tf.image.decode_jpeg(file_contents)
            image = _aspect_preserving_resize(image, image_size)
            image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
            image = tf.image.per_image_standardization(image)
            image.set_shape((image_size, image_size, 3))
            frames.append(image)
        frames_and_labels.append([frames, label])

    if shuffle:
        frames_batch, labels_batch = tf.train.shuffle_batch_join(
            frames_and_labels, batch_size=batch_size, capacity=4 * num_threads * batch_size,
            min_after_dequeue=batch_size
        )
    else:
        frames_batch, labels_batch = tf.train.batch_join(
            frames_and_labels, batch_size=batch_size,
            capacity=4 * num_threads * batch_size,
        )

    return frames_batch, labels_batch


def sample_videos(data_reader, root_folder, num_samples, num_frames):
    """
    Args:
        lines: an array contains list of images to parse
        data_reader:
        root_folder:
        num_samples:
        num_frames:
    Returns:
        - image_paths: 2D array with num_samples rows which each row contains num_frames images
        - labels: array of num_samples labels
    """
    image_paths = list()
    labels = list()
    while True:
        if len(labels) >= num_samples:
            break
        line = next(data_reader)
        video_folder, label, max_frames = line.strip().split(" ")
        max_frames = int(max_frames)
        label = int(label)
        if max_frames > num_frames:
            start_index = random.randint(0, max_frames - num_frames)
            # start_index = max(0, int(max_frames / 2 - num_frames / 2) - 1)
            frame_paths = list()
            for index in range(start_index, start_index + num_frames):
                frame_path = os.path.join(root_folder, video_folder, "%04d.jpg" % index)
                frame_paths.append(frame_path)
            image_paths.append(frame_paths)
            labels.append(label)
    return image_paths, labels


def _smallest_size_at_least(height, width, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

    height = tf.to_float(height)
    width = tf.to_float(width)
    smallest_side = tf.to_float(smallest_side)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    return new_height, new_width


def _aspect_preserving_resize(image, smallest_side):
    smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)
    shape = tf.shape(image)
    height = shape[0]
    width = shape[1]
    new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
    image = tf.expand_dims(image, 0)
    resized_image = tf.image.resize_bilinear(image, [new_height, new_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])
    return resized_image

def visualize_frame_batches(frame_batches, num_videos, num_frames):
    for i in range(num_videos):
        print("Visualize video ", i)
        video = frame_batches[i, :, :, :, :]
        for j in range(num_frames):
            frame = video[j, :, :, :]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("frame", frame)
            cv2.waitKey(100)

if __name__ == "__main__":
    num_frames = 10
    root_folder = "/home/ubuntu/datasets/ucf101/train/"
    data_reader = lines_from_file("/home/ubuntu/datasets/ucf101/train.txt", repeat=True)
    # image_paths, labels = sample_videos(data_reader, root_folder=root_folder,
    #                                     num_samples=3, num_frames=num_frames)
    #
    # for i in range(3):
    #     print(image_paths[i], labels[i])

    image_paths_placeholder = tf.placeholder(tf.string, shape=(None, num_frames), name='image_paths')
    labels_placeholder = tf.placeholder(tf.int64, shape=(None, ), name='labels')

    input_queue = data_flow_ops.FIFOQueue(capacity=10000,
                                          dtypes=[tf.string, tf.int64],
                                          shapes=[(num_frames,), ()])

    enqueue_op = input_queue.enqueue_many([image_paths_placeholder, labels_placeholder])

    batch_size = 10
    frame_batches, labels_batch = input_pipeline(input_queue, batch_size=batch_size)

    with tf.Session() as sess:
        coords = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coords)
        image_paths, labels = sample_videos(data_reader, root_folder=root_folder,
                                            num_samples=20, num_frames=num_frames)
        print("Enqueue %d videos" % len(labels))
        sess.run(enqueue_op, feed_dict={
            image_paths_placeholder: image_paths,
            labels_placeholder: labels
        })

        for i in range(2):
            print("-------")
            frame_batches_value, labels_batch_value = sess.run([frame_batches, labels_batch])
            visualize_frame_batches(frame_batches_value, num_videos=batch_size, num_frames=10)
            print(frame_batches_value.shape)
            print(labels_batch_value)

        coords.request_stop()
        coords.join(threads)
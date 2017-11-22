import tensorflow as tf
import os


def load_files(filename_queue):
    """
    Read and parse examples from data files.

    Args:
        filename: A list of string: filenames to read from

    Returns:
        uint8image: a [height, width, depth] uint8 Tensor with the image data
        label: a int32 Tensor
    """

    line_reader = tf.TextLineReader()
    key, line = line_reader.read(filename_queue)
    label, image_path = tf.decode_csv(records=line,
                                      record_defaults=[tf.constant([], dtype=tf.int32), tf.constant([], dtype=tf.string)],
                                      field_delim=' ')
    file_contents = tf.read_file(image_path)
    image = tf.image.decode_jpeg(file_contents, channels=3)

    return image, label


def input_pipeline(dataset_dir, batch_size, num_threads=8, is_training=True, shuffle=True, user_dir=None):
    if is_training:
        file_names = [os.path.join(dataset_dir, "trainval.txt")]
        if user_dir:
            file_names += [os.path.join(user_dir, "trainval.txt")]
    else:
        file_names = [os.path.join(dataset_dir, "test.txt")]
        if user_dir:
            file_names += [os.path.join(user_dir, "test.txt")]
    filename_queue = tf.train.string_input_producer(file_names)
    image, label = load_files(filename_queue)

    image = preprocessing(image, is_training)

    min_after_dequeue = 100
    capacity = min_after_dequeue + 3 * batch_size
    if shuffle:
        image_batch, label_batch = tf.train.shuffle_batch(
            [image, label], batch_size, capacity, min_after_dequeue, num_threads
        )
    else:
        image_batch, label_batch = tf.train.batch(
            [image, label], batch_size, num_threads, capacity
        )
    return image_batch, label_batch


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


def preprocessing(image, is_training=True, image_size=224, resize_side_min=256, resize_side_max=312):
    image = tf.cast(image, tf.float32)

    if is_training:
        resize_side = tf.random_uniform([], minval=resize_side_min, maxval=resize_side_max+1, dtype=tf.int32)
        resized_image = _aspect_preserving_resize(image, resize_side)

        distorted_image = tf.random_crop(resized_image, [image_size, image_size, 3])

        distorted_image = tf.image.random_flip_left_right(distorted_image)

        distorted_image = tf.image.random_brightness(distorted_image, max_delta=50)

        distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=2.0)

        return distorted_image
    else:
        resized_image = _aspect_preserving_resize(image, image_size)
        return tf.image.resize_image_with_crop_or_pad(resized_image, image_size, image_size)


import os
import tensorflow as tf
from tqdm import tqdm
from scipy.misc import imread, imsave

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir', 'data/datasets',
    'The location of Oxford IIIT Pet Dataset which contains annotations and images folders'
)

tf.app.flags.DEFINE_string(
    'target_dir', 'data/train_data',
    'The location where all the images will be stored'
)


def ensure_folder_exists(folder_path):
    """
    Create the folder at 'folder_path' if it doesn't exist
    Args:
        folder_path: the expected folder
    Returns:

    """
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    return folder_path


def read_image(image_path):
    """
    Return True if `image_path` is a readable image. Otherwise, return False
    Args:
        image_path: the path to the image
    Returns:
        image if `image_path` is a readable image. Otherwise, return None

    """
    try:
        image = imread(image_path)
        return image
    except IOError:
        print(image_path, "not readable")
    return None


def convert_data(split_name, save_label=False):
    """
    Move the given filenames to the target locations. Save list images in 'split_name'.txt in target_dir
    Args:
        split_name: The name of the dataset, either 'trainval' or 'test'
        save_label: Boolean variable indicated whether the label files will be saved
    Returns:

    """
    if split_name not in ["trainval", "test"]:
        raise ValueError("split_name is not recognized!")
    target_split_path = ensure_folder_exists(os.path.join(FLAGS.target_dir, split_name))
    output_file = open(os.path.join(FLAGS.target_dir, split_name + ".txt"), "w")

    image_folder = os.path.join(FLAGS.dataset_dir, "images")
    anno_folder = os.path.join(FLAGS.dataset_dir, "annotations")

    list_data = [line.strip() for line in open(anno_folder + "/" + split_name + ".txt")]

    class_name_idx_map = dict()
    for data in tqdm(list_data, desc=split_name):
        file_name, class_index, species, breed_id = data.split(" ")
        file_label = int(class_index) - 1

        class_name = "_".join(file_name.split("_")[0:-1])
        class_name_idx_map[class_name] = file_label

        image_path = os.path.join(image_folder, file_name + ".jpg")
        image = read_image(image_path)
        if image is not None:
            target_class_dir = ensure_folder_exists(os.path.join(target_split_path, class_name))
            target_image_path = os.path.join(target_class_dir, file_name + ".jpg")
            imsave(target_image_path, image)
            output_file.write("%s %s\n" % (file_label, target_image_path))

    if save_label:
        label_file = open(os.path.join(FLAGS.target_dir, "labels.txt"), "w")
        for class_name in sorted(class_name_idx_map, key=class_name_idx_map.get):
            label_file.write("%s\n" % class_name)


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError("You must supply the dataset directory with --dataset_dir")

    ensure_folder_exists(FLAGS.target_dir)
    convert_data("trainval", save_label=True)
    convert_data("test")


if __name__ == "__main__":
    tf.app.run()

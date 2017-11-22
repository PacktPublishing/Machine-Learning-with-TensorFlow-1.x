import av
# https://mikeboers.github.io/PyAV/installation.html
import os
import random
import tensorflow as tf
from tqdm import tqdm

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'dataset_dir', '/mnt/DATA02/Dataset/UCF101/UCF-101',
    'The folder that contains the extracted content of UCF101.rar'
)

tf.app.flags.DEFINE_string(
    'train_test_list_dir', '/mnt/DATA02/Dataset/UCF101/ucfTrainTestlist',
    'The folder that contains the extracted content of UCF101TrainTestSplits-RecognitionTask.zip'
)

tf.app.flags.DEFINE_string(
    'target_dir', '/home/ubuntu/datasets/ucf101-new',
    'The location where all the images will be stored'
)

tf.app.flags.DEFINE_integer(
    'fps', 4,
    'Framerate to export'
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


def convert_data(list_files, training=False):
    lines = []
    for txt in list_files:
        lines += [line.strip() for line in open(os.path.join(FLAGS.train_test_list_dir, txt))]
    output_name = "train"
    if training is False:
        output_name = "test"

    random.shuffle(lines)

    target_dir = ensure_folder_exists(os.path.join(FLAGS.target_dir, output_name))

    class_index = {line.split(" ")[1].strip(): int(line.split(" ")[0]) - 1 for line in open(os.path.join(FLAGS.train_test_list_dir, "classInd.txt"))}

    with open(os.path.join(FLAGS.target_dir, output_name + ".txt"), "w") as f:
        for line in tqdm(lines):
            if training:
                filename, _ = line.strip().split(" ")
            else:
                filename = line.strip()
            class_folder, video_name = filename.split("/")

            label = class_index[class_folder]
            video_name = video_name.replace(".avi", "")

            target_class_folder = ensure_folder_exists(os.path.join(target_dir, class_folder))
            target_folder = ensure_folder_exists(os.path.join(target_class_folder, video_name))

            container = av.open(os.path.join(FLAGS.dataset_dir, filename))

            frame_to_skip = int(25.0 / FLAGS.fps)
            last_frame = -1
            frame_index = 0
            for frame in container.decode(video=0):
                if last_frame < 0 or frame.index > last_frame + frame_to_skip:
                    last_frame = frame.index
                    # image = frame.to_image()
                    # target_file = os.path.join(target_folder, "%04d.jpg" % frame_index)
                    # image.save(target_file)
                    frame_index += 1
            f.write("{} {} {}\n".format("%s/%s" % (class_folder, video_name), label, frame_index))

    if training:
        with open(os.path.join(FLAGS.target_dir, "label.txt"), "w") as f:
            for class_name in sorted(class_index, key=class_index.get):
                f.write("%s\n" % class_name)


def main(_):
    if not FLAGS.dataset_dir:
        raise ValueError("You must supply the dataset directory with --dataset_dir")

    ensure_folder_exists(FLAGS.target_dir)
    convert_data(["trainlist01.txt", "trainlist02.txt", "trainlist03.txt"], training=True)
    convert_data(["testlist01.txt", "testlist02.txt", "testlist03.txt"], training=False)

if __name__ == "__main__":
    tf.app.run()


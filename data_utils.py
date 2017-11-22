import sys
import os
import numpy as np
import tensorflow as tf
import csv
import pickle
import tarfile
import zipfile as z
import threading
from scipy import ndimage
from scipy.misc import imresize, imsave

from six.moves.urllib.request import urlretrieve


MB = 1024 ** 2


def download_hook_function(block, block_size, total_size):
    if total_size != -1:
        sys.stdout.write('Downloaded: %3.3fMB of %3.3fMB\r' % (float(block * block_size) / float(MB),
                                                               float(total_size) / float(MB)))
    else:
        sys.stdout.write('Downloaded: %3.3fMB of \'unknown size\'\r' % (float(block * block_size) / float(MB)))

    sys.stdout.flush()


def download_file(file_url, output_file_dir, expected_size, FORCE=False):
    name = file_url.split('/')[-1]
    file_output_path = os.path.join(output_file_dir, name)
    print('Attempting to download ' + file_url)
    print('File output path: ' + file_output_path)
    print('Expected size: ' + str(expected_size))
    if not os.path.isdir(output_file_dir):
        os.makedirs(output_file_dir)

    if os.path.isfile(file_output_path) and os.stat(file_output_path).st_size == expected_size and not FORCE:
        print('File already downloaded completely!')
        return file_output_path
    else:
        print(' ')
        filename, _ = urlretrieve(file_url, file_output_path, download_hook_function)
        print(' ')
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_size:
            print('Found and verified', filename)
        else:
            raise Exception('Could not download ' + filename)
        return filename


def extract_file(input_file, output_dir, FORCE=False):
    if os.path.isdir(output_dir) and not FORCE:
        print('%s already extracted to %s' % (input_file, output_dir))
        directories = [x for x in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, x))]
        return output_dir + "/" + directories[0]
    else:
        tar = tarfile.open(input_file)
        sys.stdout.flush()
        print('Started extracting:\n%s\nto:\n%s' % (input_file, output_dir))
        tar.extractall(output_dir)
        print('Finished extracting:\n%s\nto:\n%s' % (input_file, output_dir))
        tar.close()
        directories = [x for x in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, x))]
        return output_dir + "/" + directories[0]


def load_class(folder, image_size, pixel_depth):
    image_files = os.listdir(folder)
    num_of_images = len(image_files)
    dataset = np.ndarray(shape=(num_of_images, image_size, image_size),
                         dtype=np.float32)
    image_index = 0
    print('Started loading images from: ' + folder)
    for index, image in enumerate(image_files):

        sys.stdout.write('Loading image %d of %d\r' % (index + 1, num_of_images))
        sys.stdout.flush()

        image_file = os.path.join(folder, image)

        try:
            image_data = (ndimage.imread(image_file).astype(float) -
                          pixel_depth / 2) / pixel_depth
            if image_data.shape != (image_size, image_size):
                raise Exception('Unexpected image shape: %s' % str(image_data.shape))
            dataset[image_index, :, :] = image_data
            image_index += 1
        except IOError as e:
            print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    print('Finished loading data from: ' + folder)

    return dataset[0:image_index, :, :]


def make_pickles(input_folder, output_dir, image_size, image_depth, FORCE=False):
    directories = sorted([x for x in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, x))])
    pickle_files = [os.path.join(output_dir, x + '.pickle') for x in directories]

    for index, pickle_file in enumerate(pickle_files):

        if os.path.isfile(pickle_file) and not FORCE:
            print('\tPickle already exists: %s' % (pickle_file))
        else:
            folder_path = os.path.join(input_folder, directories[index])
            print('\tLoading from folder: ' + folder_path)
            data = load_class(folder_path, image_size, image_depth)

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)

            print('\tStarted pickling: ' + directories[index])
            try:
                with open(pickle_file, 'wb') as f:
                    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            except Exception as e:
                print('Unable to save data to', pickle_file, ':', e)
            print('Finished pickling: ' + directories[index])

    return pickle_files


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels


def make_arrays(nb_rows, img_size):
    if nb_rows:
        dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
        labels = np.ndarray(nb_rows, dtype=np.int32)
    else:
        dataset, labels = None, None
    return dataset, labels


def reformat(data, image_size, num_of_channels, num_of_classes, flatten=True):
    if flatten:
        data.train_dataset = data.train_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
        data.valid_dataset = data.valid_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
        data.test_dataset = data.test_dataset.reshape((-1, image_size * image_size * num_of_channels)).astype(np.float32)
    else:
        data.train_dataset = data.train_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)
        data.valid_dataset = data.valid_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)
        data.test_dataset = data.test_dataset.reshape((-1, image_size, image_size, num_of_channels)).astype(np.float32)

    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    data.train_labels = (np.arange(num_of_classes) == data.train_labels[:, None]).astype(np.float32)
    data.valid_labels = (np.arange(num_of_classes) == data.valid_labels[:, None]).astype(np.float32)
    data.test_labels = (np.arange(num_of_classes) == data.test_labels[:, None]).astype(np.float32)

    return data


def merge_datasets(pickle_files, image_size, train_size, valid_size=0):
    num_classes = len(pickle_files)
    valid_dataset, valid_labels = make_arrays(valid_size, image_size)
    train_dataset, train_labels = make_arrays(train_size, image_size)
    vsize_per_class = valid_size // num_classes
    tsize_per_class = train_size // num_classes

    start_v, start_t = 0, 0
    end_v, end_t = vsize_per_class, tsize_per_class
    end_l = vsize_per_class + tsize_per_class
    for label, pickle_file in enumerate(pickle_files):
        try:
            with open(pickle_file, 'rb') as f:
                letter_set = pickle.load(f)
                # let's shuffle the letters to have random validation and training set
                np.random.shuffle(letter_set)
                if valid_dataset is not None:
                    valid_letter = letter_set[:vsize_per_class, :, :]
                    valid_dataset[start_v:end_v, :, :] = valid_letter
                    valid_labels[start_v:end_v] = label
                    start_v += vsize_per_class
                    end_v += vsize_per_class

                train_letter = letter_set[vsize_per_class:end_l, :, :]
                train_dataset[start_t:end_t, :, :] = train_letter
                train_labels[start_t:end_t] = label
                start_t += tsize_per_class
                end_t += tsize_per_class
        except Exception as e:
            print('Unable to process data from', pickle_file, ':', e)
            raise

    return valid_dataset, valid_labels, train_dataset, train_labels


def pickle_whole(train_pickle_files, test_pickle_files, image_size,
                 train_size, valid_size, test_size, output_file_path, FORCE=False):
    if os.path.isfile(output_file_path) and not FORCE:
        print('Pickle file: %s already exist' % (output_file_path))

        with open(output_file_path, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    else:
        print('Merging train, valid data')
        valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
            train_pickle_files, image_size, train_size, valid_size)
        print('Merging test data')
        _, _, test_dataset, test_labels = merge_datasets(test_pickle_files, image_size, test_size)
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)

        train_dataset, train_labels = randomize(train_dataset, train_labels)
        test_dataset, test_labels = randomize(test_dataset, test_labels)
        valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)
        try:
            f = open(output_file_path, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', output_file_path, ':', e)
            raise

        statinfo = os.stat(output_file_path)
        print('Compressed pickle size:', statinfo.st_size)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def load_cifar_10_pickle(pickle_file, image_depth):
    fo = open(pickle_file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return ((dict['data'].astype(float) - image_depth / 2) / (image_depth)), dict['labels']


def load_cifar_10_from_pickles(train_pickle_files, test_pickle_files, pickle_batch_size, image_size, image_depth,
                               num_of_channels):
    all_train_data = np.ndarray(shape=(pickle_batch_size * len(train_pickle_files),
                                       image_size * image_size * num_of_channels),
                                dtype=np.float32)

    all_train_labels = np.ndarray(shape=pickle_batch_size * len(train_pickle_files), dtype=object)

    all_test_data = np.ndarray(shape=(pickle_batch_size * len(test_pickle_files),
                                      image_size * image_size * num_of_channels),
                               dtype=np.float32)
    all_test_labels = np.ndarray(shape=pickle_batch_size * len(test_pickle_files), dtype=object)

    print('Started loading training data')
    for index, train_pickle_file in enumerate(train_pickle_files):
        all_train_data[index * pickle_batch_size: (index + 1) * pickle_batch_size, :], \
        all_train_labels[index * pickle_batch_size: (index + 1) * pickle_batch_size] = \
            load_cifar_10_pickle(train_pickle_file, image_depth)
    print('Finished loading training data\n')

    print('Started loading testing data')
    for index, test_pickle_file in enumerate(test_pickle_files):
        all_test_data[index * pickle_batch_size: (index + 1) * pickle_batch_size, :], \
        all_test_labels[index * pickle_batch_size: (index + 1) * pickle_batch_size] = \
            load_cifar_10_pickle(test_pickle_file, image_depth)
    print('Finished loading testing data')

    return all_train_data, all_train_labels, all_test_data, all_test_labels


def pickle_cifar_10(all_train_data, all_train_labels, all_test_data, all_test_labels,
                    train_size, valid_size, test_size, output_file_path, FORCE=False):
    if os.path.isfile(output_file_path) and not FORCE:
        print('\tPickle file already exists: %s' % output_file_path)

        with open(output_file_path, 'rb') as f:
            save = pickle.load(f)
            train_dataset = save['train_dataset']
            train_labels = save['train_labels']
            valid_dataset = save['valid_dataset']
            valid_labels = save['valid_labels']
            test_dataset = save['test_dataset']
            test_labels = save['test_labels']
            del save  # hint to help gc free up memory
            print('Training set', train_dataset.shape, train_labels.shape)
            print('Validation set', valid_dataset.shape, valid_labels.shape)
            print('Test set', test_dataset.shape, test_labels.shape)

        return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
    else:
        train_dataset = all_train_data[0:train_size]
        train_labels = all_train_labels[0:train_size]
        valid_dataset = all_train_data[train_size:train_size + valid_size]
        valid_labels = all_train_labels[train_size:train_size + valid_size]
        test_dataset = all_test_data[0:test_size]
        test_labels = all_test_labels[0:test_size]

        try:
            f = open(output_file_path, 'wb')
            save = {
                'train_dataset': train_dataset,
                'train_labels': train_labels,
                'valid_dataset': valid_dataset,
                'valid_labels': valid_labels,
                'test_dataset': test_dataset,
                'test_labels': test_labels,
            }
            pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
            f.close()
        except Exception as e:
            print('Unable to save data to', output_file_path, ':', e)
            raise

        statinfo = os.stat(output_file_path)
        print('Compressed pickle size:', statinfo.st_size)

    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


def check_file_status(file_path, expected_size, error_message, close=True):
    file_size = os.stat(file_path).st_size
    if file_size == expected_size:
        print("File status ({}): OK".format(file_path))
        return True
    else:
        print("File status ({}): CORRUPTED. Expected size: {}, found: {}".format(file_path, expected_size, file_size))
        print(error_message)
        if close:
            exit(-1)
        else:
            return False


def check_folder_status(folder_path, expected_num_of_files, success_message, error_message, close=True):
    num_of_files_found = 0

    for root, dirs, files in os.walk(folder_path):
        num_of_files_found += len(files)

    if num_of_files_found == expected_num_of_files:
        print(success_message)
        return True
    else:
        print(error_message)
        if close:
            exit(-1)
        else:
            return False


def crop_black_borders(image, threshold=0):
    """Crops any edges below or equal to threshold

    Crops blank image to 1x1.

    Returns cropped image.

    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def prepare_not_mnist_dataset(root_dir="."):
    print('Started preparing notMNIST dataset')

    image_size = 28
    image_depth = 255

    training_set_url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz'
    test_set_url = 'http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz'

    train_download_size = 247336696
    test_download_size = 8458043

    train_size = 200000
    valid_size = 10000
    test_size = 10000

    num_of_classes = 10
    num_of_channels = 1

    dataset_path = os.path.realpath(os.path.join(root_dir, "datasets", "notMNIST"))
    train_path = os.path.join(dataset_path, "train")
    test_path = os.path.join(dataset_path, "test")

    train_file_path = download_file(training_set_url, dataset_path, train_download_size)
    test_file_path = download_file(test_set_url, dataset_path, test_download_size)

    train_extracted_folder = extract_file(train_file_path, train_path)
    test_extracted_folder = extract_file(test_file_path, test_path)

    print('Started loading training data')
    train_pickle_files = make_pickles(train_extracted_folder, train_path, image_size, image_depth)
    print('Finished loading training data\n')

    print('Started loading testing data')
    test_pickle_files = make_pickles(test_extracted_folder, test_path, image_size, image_depth)
    print('Finished loading testing data')

    print('Started pickling final dataset')
    train_dataset, train_labels, valid_dataset, valid_labels, \
    test_dataset, test_labels = pickle_whole(train_pickle_files, test_pickle_files, image_size, train_size, valid_size,
                                             test_size, os.path.join(dataset_path, 'notMNIST.pickle'))
    print('Finished pickling final dataset')

    print('Finished preparing notMNIST dataset')

    def not_mnist(): pass

    not_mnist.train_dataset = train_dataset
    not_mnist.train_labels = train_labels
    not_mnist.valid_dataset = valid_dataset
    not_mnist.valid_labels = valid_labels
    not_mnist.test_dataset = test_dataset
    not_mnist.test_labels = test_labels

    return not_mnist, image_size, num_of_classes, num_of_channels


def prepare_cifar_10_dataset():
    print('Started preparing CIFAR-10 dataset')

    image_size = 32
    image_depth = 255
    cifar_dataset_url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    dataset_size = 170498071
    train_size = 45000
    valid_size = 5000
    test_size = 10000
    num_of_classes = 10
    num_of_channels = 3
    pickle_batch_size = 10000

    dataset_path = download_file(cifar_dataset_url,
                                 os.path.realpath('../../datasets/CIFAR-10'), dataset_size)

    dataset_extracted_folder = extract_file(dataset_path, os.path.realpath('../../datasets/CIFAR-10/data'))

    train_pickle_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
                          'data_batch_5']
    train_pickle_files = [dataset_extracted_folder + '/' + x for x in train_pickle_files]

    test_pickle_files = ['test_batch']
    test_pickle_files = [dataset_extracted_folder + '/' + x for x in test_pickle_files]

    print('Started loading CIFAR-10 dataset')
    all_train_data, all_train_labels, all_test_data, all_test_labels = load_cifar_10_from_pickles(train_pickle_files,
                                                                                                  test_pickle_files,
                                                                                                  pickle_batch_size,
                                                                                                  image_size,
                                                                                                  image_depth,
                                                                                                  num_of_channels)
    print('Finished loading CIFAR-10 dataset')

    print('Started pickling final dataset')
    train_dataset, train_labels, valid_dataset, valid_labels, \
    test_dataset, test_labels = pickle_cifar_10(all_train_data, all_train_labels, all_test_data, all_test_labels,
                                                train_size, valid_size, test_size,
                                                os.path.realpath('../../datasets/CIFAR-10/CIFAR-10.pickle'), True)
    print('Finished pickling final dataset')

    print('Finished preparing CIFAR-10 dataset')

    def cifar_10(): pass

    cifar_10.train_dataset = train_dataset
    cifar_10.train_labels = train_labels
    cifar_10.valid_dataset = valid_dataset
    cifar_10.valid_labels = valid_labels
    cifar_10.test_dataset = test_dataset
    cifar_10.test_labels = test_labels

    return cifar_10, image_size, num_of_classes, num_of_channels


def prepare_dr_dataset(dataset_dir):
    num_of_processing_threads = 16

    dr_dataset_base_path = os.path.realpath(dataset_dir)

    unique_labels_file_path = os.path.join(dr_dataset_base_path, "unique_labels_file.txt")

    processed_images_folder = os.path.join(dr_dataset_base_path, "processed_images")
    num_of_processed_images = 35126

    train_processed_images_folder = os.path.join(processed_images_folder, "train")
    validation_processed_images_folder = os.path.join(processed_images_folder, "validation")

    num_of_training_images = 30000

    raw_images_folder = os.path.join(dr_dataset_base_path, "train")

    train_labels_csv_path = os.path.join(dr_dataset_base_path, "trainLabels.csv")

    def process_images_batch(thread_index, files, labels, subset):

        num_of_files = len(files)

        for index, file_and_label in enumerate(zip(files, labels)):
            file = file_and_label[0] + '.jpeg'
            label = file_and_label[1]

            input_file = os.path.join(raw_images_folder, file)
            output_file = os.path.join(processed_images_folder, subset, str(label), file)

            image = ndimage.imread(input_file)
            cropped_image = crop_black_borders(image, 10)
            resized_cropped_image = imresize(cropped_image, (299, 299, 3), interp="bicubic")
            imsave(output_file, resized_cropped_image)

            if index % 10 == 0:
                print("(Thread {}): Files processed {} out of {}".format(thread_index, index, num_of_files))

    def process_images(files, labels, subset):

        # Break all images into batches with a [ranges[i][0], ranges[i][1]].
        spacing = np.linspace(0, len(files), num_of_processing_threads + 1).astype(np.int)
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])

        # Create a mechanism for monitoring when all threads are finished.
        coord = tf.train.Coordinator()

        threads = []
        for thread_index in range(len(ranges)):
            args = (thread_index, files[ranges[thread_index][0]:ranges[thread_index][1]],
                    labels[ranges[thread_index][0]:ranges[thread_index][1]],
                    subset)
            t = threading.Thread(target=process_images_batch, args=args)
            t.start()
            threads.append(t)

        # Wait for all the threads to terminate.
        coord.join(threads)

    def process_training_and_validation_images():
        train_files = []
        train_labels = []

        validation_files = []
        validation_labels = []

        with open(train_labels_csv_path) as csvfile:
            reader = csv.DictReader(csvfile)
            for index, row in enumerate(reader):
                if index < num_of_training_images:
                    train_files.extend([row['image'].strip()])
                    train_labels.extend([int(row['level'].strip())])
                else:
                    validation_files.extend([row['image'].strip()])
                    validation_labels.extend([int(row['level'].strip())])

        if not os.path.isdir(processed_images_folder):
            os.mkdir(processed_images_folder)

        if not os.path.isdir(train_processed_images_folder):
            os.mkdir(train_processed_images_folder)

        if not os.path.isdir(validation_processed_images_folder):
            os.mkdir(validation_processed_images_folder)

        for directory_index in range(5):
            train_directory_path = os.path.join(train_processed_images_folder, str(directory_index))
            valid_directory_path = os.path.join(validation_processed_images_folder, str(directory_index))

            if not os.path.isdir(train_directory_path):
                os.mkdir(train_directory_path)

            if not os.path.isdir(valid_directory_path):
                os.mkdir(valid_directory_path)

        print("Processing training files...")
        process_images(train_files, train_labels, "train")
        print("Done!")

        print("Processing validation files...")
        process_images(validation_files, validation_labels, "validation")
        print("Done!")

        print("Making unique labels file...")
        with open(unique_labels_file_path, 'w') as unique_labels_file:
            unique_labels = ""
            for index in range(5):
                unique_labels += "{}\n".format(index)
            unique_labels_file.write(unique_labels)

        status = check_folder_status(processed_images_folder, num_of_processed_images,
                                     "All processed images are present in place",
                                     "Couldn't complete the image processing of training and validation files.")

        return status

    process_training_and_validation_images()
    return

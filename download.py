from utils import *
output_dir = "data"
# train_tar = download_file('http://yaroslavvb.com/upload/notMNIST/notMNIST_large.tar.gz', 247336696, output_dir)
# test_tar = download_file('http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz', 8458043, output_dir)
train_tar = "data/notMNIST_large.tar.gz"
test_tar = "data/notMNIST_small.tar.gz"
print ('Train set stored in: ' + train_tar)
print ('Test set stored in: ' + test_tar)
train_folder = extract_file(train_tar, os.path.join(output_dir, "train"))
test_folder = extract_file(test_tar, os.path.join(output_dir, "test"))
print ('Train file set stored in: ' + train_folder)
print ('Test file set stored in: ' + test_folder)
train_pickle = make_pickle(train_folder)
print("Train pickle", train_pickle)
test_pickle = make_pickle(test_folder)
print("Test pickle", test_pickle)

train_pickle = ['data/train/notMNIST_large/D.pickle', 'data/train/notMNIST_large/I.pickle', 'data/train/notMNIST_large/C.pickle', 'data/train/notMNIST_large/B.pickle', 'data/train/notMNIST_large/E.pickle', 'data/train/notMNIST_large/H.pickle', 'data/train/notMNIST_large/J.pickle', 'data/train/notMNIST_large/A.pickle', 'data/train/notMNIST_large/G.pickle', 'data/train/notMNIST_large/F.pickle']
test_pickle = ['data/test/notMNIST_small/D.pickle', 'data/test/notMNIST_small/I.pickle', 'data/test/notMNIST_small/C.pickle', 'data/test/notMNIST_small/B.pickle', 'data/test/notMNIST_small/E.pickle', 'data/test/notMNIST_small/H.pickle', 'data/test/notMNIST_small/J.pickle', 'data/test/notMNIST_small/A.pickle', 'data/test/notMNIST_small/G.pickle', 'data/test/notMNIST_small/F.pickle']
train_size = 200000
valid_size = 10000
test_size = 10000

valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
    train_pickle, train_size, valid_size)
_, _, test_dataset, test_labels = merge_datasets(test_pickle, test_size)

print('Training:', train_dataset.shape, train_labels.shape)
print('Validation:', valid_dataset.shape, valid_labels.shape)
print('Testing:', test_dataset.shape, test_labels.shape)

train_dataset, train_labels = randomize(train_dataset, train_labels)
test_dataset, test_labels = randomize(test_dataset, test_labels)
valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

pickle_file = os.path.join(output_dir, 'notMNIST.pickle')

try:
    f = open(pickle_file, 'wb')
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
    print('Unable to save data to', pickle_file, ':', e)
    raise

statinfo = os.stat(pickle_file)
print('Compressed pickle size:', statinfo.st_size)
print("Final dataset is stored at", pickle_file)


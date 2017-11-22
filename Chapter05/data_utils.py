import os, sys, tarfile
from six.moves.urllib.request import urlretrieve

import gzip
import os
import re

from tensorflow.python.platform import gfile

MB = 1024 ** 2

# Special symbols
_PAD = b"_PAD"  # For padding empty spaces in buckets
_GO = b"_GO"  # Initial symbol for the decoder input
_EOS = b"_EOS"  # End of sentense
_UNK = b"_UNK"  # Symbol for unknown vocabulary
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

# IDs for the above mentioned symbols
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions for tokeninzing words and digits
_WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")  # RE for words
_DIGIT_RE = re.compile(br"\d")  # RE for digits


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
        print(statinfo.st_size)
        if statinfo.st_size == expected_size:
            print('Found and verified', filename)
        else:
            raise Exception('Could not download ' + filename)
        return filename


def extract_file(input_file, output_dir_or_file, TYPE='tar', IS_SUB=True, FORCE=False):
    if (os.path.isdir(output_dir_or_file) or os.path.isfile(output_dir_or_file)) and not FORCE:
        print('%s already extracted to %s' % (input_file, output_dir_or_file))
        if IS_SUB:
            directories = [x for x in os.listdir(output_dir_or_file)
                           if os.path.isdir(os.path.join(output_dir_or_file, x))]
            return output_dir_or_file + "/" + directories[0]
        else:
            return output_dir_or_file
    else:
        print('Started extracting %s to %s' % (input_file, output_dir_or_file))
        if TYPE == 'tar':
            tar = tarfile.open(input_file)
            sys.stdout.flush()
            tar.extractall(output_dir_or_file)
            tar.close()
        elif TYPE == 'gz':
            with gzip.open(input_file, "rb") as gz_file:
                with open(output_dir_or_file, "wb") as new_file:
                    for line in gz_file:
                        new_file.write(line)
        else:
            print('Could not identify compress file type \'%s\'' % TYPE)
            return None
        print('Finished extracting %s to %s' % (input_file, output_dir_or_file))
        if IS_SUB:
            directories = [x for x in os.listdir(output_dir_or_file) if
                           os.path.isdir(os.path.join(output_dir_or_file, x))]
            return output_dir_or_file + "/" + directories[0]
        else:
            return output_dir_or_file


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=True):
    """Create vocabulary file (if it does not exist yet) from data file.
  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.
  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
    if not gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with gfile.GFile(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.
  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].
  Args:
    vocabulary_path: path to the file containing the vocabulary.
  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).
  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
    if gfile.Exists(vocabulary_path):
        rev_vocab = []
        with gfile.GFile(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          tokenizer=None, normalize_digits=True):
    """Convert a string to list of integers representing token-ids.
  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].
  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  Returns:
    a list of integers, the token-ids for the sentence.
  """

    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.
  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.
  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
    if not gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print("  tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                                      normalize_digits)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_wmt_dataset(tokenizer=None):
    vocab_size = 40000

    # URLs for WMT data.
    _WMT_ENFR_TRAIN_URL = "http://www.statmt.org/wmt10/training-giga-fren.tar"
    _WMT_ENFR_DEV_URL = "http://www.statmt.org/wmt15/dev-v2.tgz"

    # Expected number of bytes for the above two file downloads
    _WMT_ENFR_TRAIN_SIZE = 2595102720
    _WMT_ENFR_DEV_SIZE = 21393583

    train_file_path = download_file(_WMT_ENFR_TRAIN_URL,
                                    os.path.realpath('../datasets/WMT'), _WMT_ENFR_TRAIN_SIZE)

    dev_file_path = download_file(_WMT_ENFR_DEV_URL,
                                  os.path.realpath('../datasets/WMT'), _WMT_ENFR_DEV_SIZE)

    train_extracted_folder = extract_file(train_file_path, os.path.realpath('../datasets/WMT/train'), IS_SUB=False)
    dev_extracted_folder = extract_file(dev_file_path, os.path.realpath('../datasets/WMT'), IS_SUB=False,
                                        FORCE=True)

    train_sub_gzip_files = ['giga-fren.release2.fixed.fr.gz', 'giga-fren.release2.fixed.en.gz']
    train_sub_gzip_files = [train_extracted_folder + '/' + x for x in train_sub_gzip_files]

    if not os.path.exists(train_extracted_folder + '/data'):
        os.makedirs(train_extracted_folder + '/data')

    train_sub_extracted_files = [None] * 2
    for index, train_sub_gzip_file in enumerate(train_sub_gzip_files):
        train_sub_extracted_files[index] = extract_file(train_sub_gzip_file,
                                                        train_extracted_folder + '/data/' +
                                                        train_sub_gzip_file.split('/')[-1].split('.gz')[0],
                                                        TYPE='gz',
                                                        IS_SUB=False)

    vocab_paths = [None] * 2
    token_paths = [None] * 2

    # Create vocabularies of the appropriate sizes and tokenizing the vocabulary
    for index, train_sub_extracted_file in enumerate(train_sub_extracted_files):
        type = train_sub_extracted_file.split('.')[-1]
        vocab_paths[index] = "%s%s.%s" % (train_extracted_folder + '/data/', 'vocab%d' % vocab_size, type)
        token_paths[index] = "%s%s.%s" % (train_extracted_folder + '/data/', 'token%d' % vocab_size, type)
        create_vocabulary(vocab_paths[index], train_sub_extracted_files[index], vocab_size, tokenizer)
        data_to_token_ids(train_sub_extracted_files[index], token_paths[index], vocab_paths[index], tokenizer)

    dev_sub_required_files = ['dev/newstest2013.fr', 'dev/newstest2013.en']
    dev_sub_required_files = [dev_extracted_folder + '/' + x for x in dev_sub_required_files]

    if not os.path.exists(dev_extracted_folder + '/dev/data'):
        os.makedirs(dev_extracted_folder + '/dev/data')

    dev_token_paths = [None] * 2
    for index, dev_sub_required_file in enumerate(dev_sub_required_files):
        type = dev_sub_required_file.split('.')[-1]
        dev_token_paths[index] = "%s%s.%s" % (dev_extracted_folder + '/dev/data/', 'token%d' % vocab_size, type)
        data_to_token_ids(dev_sub_required_files[index], dev_token_paths[index], vocab_paths[index], tokenizer)

    def wmt(): pass

    wmt.en_train_ids_path = token_paths[1]
    wmt.fr_train_ids_path = token_paths[0]
    wmt.en_dev_ids_path = dev_token_paths[1]
    wmt.fr_dev_ids_path = dev_token_paths[0]
    wmt.en_vocab_path = vocab_paths[1]
    wmt.fr_vocab_path = vocab_paths[0]

    return wmt

prepare_wmt_dataset()

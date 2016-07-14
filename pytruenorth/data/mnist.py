import gzip
import numpy as np

from .dataset import DataSet, maybe_download, dense_to_one_hot

NAME = 'MNIST'
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def extract_data(filename, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    """
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data / PIXEL_DEPTH
        data = data.reshape(num_images, IMAGE_SIZE * IMAGE_SIZE)
        return data


def extract_labels(filename, num_images):
    """Extract the labels into a vector of int64 label IDs."""
    print('Extracting', filename)
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return dense_to_one_hot(labels, NUM_LABELS)


def load(validation_size=1000):
    class DataSets(object):
        pass

    dataset = DataSets()

    # Get the data.
    train_data_filename = maybe_download(SOURCE_URL, 'train-images-idx3-ubyte.gz', NAME)
    train_labels_filename = maybe_download(SOURCE_URL, 'train-labels-idx1-ubyte.gz', NAME)
    test_data_filename = maybe_download(SOURCE_URL, 't10k-images-idx3-ubyte.gz', NAME)
    test_labels_filename = maybe_download(SOURCE_URL, 't10k-labels-idx1-ubyte.gz', NAME)

    # Extract it into numpy arrays.
    train_data = extract_data(train_data_filename, 60000)
    train_labels = extract_labels(train_labels_filename, 60000)
    test_data = extract_data(test_data_filename, 10000)
    test_labels = extract_labels(test_labels_filename, 10000)

    idx = np.arange(len(train_data))
    np.random.permutation(idx)

    validation_data = train_data[idx[:validation_size], :]
    validation_labels = train_labels[idx[:validation_size], :]
    train_data = train_data[idx[validation_size:], :]
    train_labels = train_labels[idx[validation_size:], :]

    dataset.train = DataSet(train_data, train_labels)
    dataset.validation = DataSet(validation_data, validation_labels)
    dataset.test = DataSet(test_data, test_labels)

    return dataset

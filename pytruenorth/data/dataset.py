import os
import numpy as np
from urllib import request

from . import CACHE_DIR


def maybe_download(source_url, filename, destdir):
    """Download the data from Yann's website, unless it's already here."""
    destdir = os.path.join(CACHE_DIR, destdir)
    if not os.path.exists(destdir):
        os.makedirs(destdir, mode=0o755)
    filepath = os.path.join(destdir, filename)
    if not os.path.exists(filepath):
        filepath, _ = request.urlretrieve(source_url + filename, filepath)
        with open(filepath) as f:
            size = f.seek(0, 2)
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self, data, labels, dtype=np.float32):
        """Construct a DataSet.
        """
        assert data.shape[0] == labels.shape[0], (
            'images.shape: %s labels.shape: %s' % (data.shape,
                                                   labels.shape))
        data = data.astype(dtype)

        self._size = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    def __len__(self):
        return self._size

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._size:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._size)
            np.random.shuffle(perm)
            self._data = self._data[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._size
        end = self._index_in_epoch
        return self._data[start:end], self._labels[start:end]

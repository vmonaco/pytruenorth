import numpy as np
import pandas as pd

from .dataset import DataSet, maybe_download, dense_to_one_hot

NAME = 'BIOSIG'
FILENAME = 'biosig.csv'
SOURCE_URL = ''

NUM_CLASSES = 2


def load(ratio_train=0.5, validation_size=50, target='gender'):
    class DataSets(object):
        pass

    dataset = DataSets()

    # Get the data.
    data_filename = maybe_download(SOURCE_URL, FILENAME, NAME)
    df = pd.read_csv(data_filename)

    df = df.groupby(['user','session']).filter(lambda x: x['keyname'].iloc[0] == 'l')

    def duration_latency(x):
        d = (x['timerelease'] - x['timepress']).values
        tau = x['timepress'].diff().dropna().values

        f = np.empty((d.size + tau.size,), dtype=np.float32)
        f[0::2] = d
        f[1::2] = tau
        return pd.Series(f)

    df = df.groupby(['user', 'session', target]).apply(duration_latency)
    df = df.reset_index()
    df['user'] = df[target]
    df = df.set_index(['user','session']).drop(target, axis=1)

    # Tile over 256 inputs
    df = pd.concat([df]*7 + [df[df.columns[:25]]], axis=1)

    df = df.reset_index()
    unique_labels = np.unique(df['user'].values)
    mapping = dict(zip(unique_labels, range(len(unique_labels))))
    df['user'] = np.array(list(map(lambda x: mapping[x], df['user'].values)))
    df = df.set_index(['user', 'session'])

    lower = df.mean(axis=0) - 2 * df.std(axis=0)
    upper = df.mean(axis=0) + 2 * df.std(axis=0)
    df = (df - lower) / (upper - lower)
    df[df < 0] = 0
    df[df > 1] = 1

    train_data = df.groupby(level=0).apply(lambda x: x[:int(ratio_train * len(x))]).reset_index(level=0, drop=True)
    test_data = df.groupby(level=0).apply(lambda x: x[int(ratio_train * len(x)):]).reset_index(level=0, drop=True)

    validation_data = train_data.groupby(level=0).apply(lambda x: x[-validation_size:]).reset_index(level=0, drop=True)
    train_data = train_data.groupby(level=0).apply(lambda x: x[:-validation_size]).reset_index(level=0, drop=True)

    train_idx = np.random.permutation(np.arange(300))
    validation_idx = np.random.permutation(np.arange(100))
    test_idx = np.random.permutation(np.arange(300))

    dataset.train = DataSet(train_data.values[train_idx],
                            dense_to_one_hot(train_data.index.get_level_values(0).values[train_idx], NUM_CLASSES))
    dataset.validation = DataSet(validation_data.values[validation_idx],
                                 dense_to_one_hot(validation_data.index.get_level_values(0).values[validation_idx], NUM_CLASSES))
    dataset.test = DataSet(test_data.values[test_idx],
                           dense_to_one_hot(test_data.index.get_level_values(0).values[test_idx], NUM_CLASSES))

    return dataset

import numpy as np
import pandas as pd

from .dataset import DataSet, maybe_download, dense_to_one_hot

NAME = 'CMU'
FILENAME = 'DSL-StrongPasswordData.csv'
SOURCE_URL = 'http://www.cs.cmu.edu/~keystroke/'

FEATURE_SIZE = 21
NUM_CLASSES = 10


def load(ratio_train=0.5, validation_size=10):
    class DataSets(object):
        pass

    dataset = DataSets()

    # Get the data.
    data_filename = maybe_download(SOURCE_URL, FILENAME, NAME)
    df = pd.read_csv(data_filename, index_col=[0, 1, 2])
    session = df.index.get_level_values(1) * 100 + df.index.get_level_values(2)
    df = df.reset_index(level=[1, 2], drop=True)
    df = df.set_index(session, append=True)
    df.index.names = ['user', 'session']

    # Reduce to 10 users
    df = df[df.index.get_level_values(0).isin(df.index.levels[0].unique()[:10])]
    # df = df[[c for c in df.columns if 'UD' not in c]]

    # Tile over 256 inputs
    df = pd.concat([df]*8 + [df[df.columns[:8]]], axis=1)

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

    dataset.train = DataSet(train_data.values,
                            dense_to_one_hot(train_data.index.get_level_values(0).values, NUM_CLASSES))
    dataset.validation = DataSet(validation_data.values,
                                 dense_to_one_hot(validation_data.index.get_level_values(0).values, NUM_CLASSES))
    dataset.test = DataSet(test_data.values, dense_to_one_hot(test_data.index.get_level_values(0).values, NUM_CLASSES))

    return dataset

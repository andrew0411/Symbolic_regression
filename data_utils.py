import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_DATA_DIR = Path(__file__).resolve().parent / 'dataset'


def resolve_data_dir(data_dir=None):
    '''
    Resolve the dataset folder: data_dir argument > $DATA_ROOT/symbolic_regression (if it exists) > dataset/ in this repository
    '''
    if data_dir is not None:
        return Path(data_dir)
    data_root = os.environ.get('DATA_ROOT')
    if data_root and (Path(data_root) / 'symbolic_regression').is_dir():
        return Path(data_root) / 'symbolic_regression'
    return REPO_DATA_DIR


def load_split(name, target='Heat of formation', index_col='Material',
               test_size=0.2, random_state=42, data_dir=None):
    '''
    Read a CSV, split it into train/test and standardize the features. The scaler is fit on train only; test is only transformed

    Parameters:
        name (str) -- dataset name, e.g. 'high', 'mid', 'low'
        target (str) -- column to predict
        index_col (str) -- column used as the index (excluded from the features)
        test_size (float) -- fraction of samples in the test set
        random_state (int) -- split seed
        data_dir (str or Path, optional) -- dataset folder. If None, resolve_data_dir() decides

    Returns:
        x_tr, x_ts, y_tr, y_ts, feature_names
    '''
    path = resolve_data_dir(data_dir) / f'{name}.csv'
    df = pd.read_csv(path).set_index(index_col)
    y = df.pop(target)
    X = df

    x_tr, x_ts, y_tr, y_ts = train_test_split(X, y, test_size=test_size,
                                              random_state=random_state)

    sc = StandardScaler()
    x_tr = sc.fit_transform(x_tr)
    x_ts = sc.transform(x_ts)

    return x_tr, x_ts, y_tr, y_ts, list(X.columns)

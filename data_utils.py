import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

REPO_DATA_DIR = Path(__file__).resolve().parent / 'dataset'


def resolve_data_dir(data_dir=None):
    '''
    Dataset 폴더를 정한다. 우선순위: data_dir 인자 > $DATA_ROOT/symbolic_regression (폴더가 있을 때) > repo의 dataset/
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
    CSV를 읽어 train/test로 나누고 표준화한다. Scaler는 train에만 fit하고 test에는 transform만 한다

    Parameters:
        name (str) -- dataset 이름, 예: 'high', 'mid', 'low'
        target (str) -- 예측할 column
        index_col (str) -- index로 쓸 column (feature에서 제외됨)
        test_size (float) -- test 비율
        random_state (int) -- split seed
        data_dir (str or Path, optional) -- dataset 폴더. None이면 resolve_data_dir()를 따른다

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

import numpy as np

from data_utils import load_split


def test_scaler_is_fit_on_train_only():
    x_tr, x_ts, y_tr, y_ts, feature_names = load_split('high')
    assert x_tr.shape[1] == x_ts.shape[1] == len(feature_names) == 85
    assert len(x_tr) == len(y_tr) and len(x_ts) == len(y_ts)
    np.testing.assert_allclose(x_tr.mean(axis=0), 0., atol=1e-8)
    # test는 train 통계로 변환되므로 평균이 0이 아니다 (test에 다시 fit하던 버그에서는 0)
    assert not np.allclose(x_ts.mean(axis=0), 0., atol=1e-8)


def test_split_is_reproducible():
    first = load_split('low')
    second = load_split('low')
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])

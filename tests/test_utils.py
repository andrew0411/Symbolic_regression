import numpy as np

from gp.utils import _partition_estimators, check_random_state


def test_partition_estimators():
    # fit() used to fail here on NumPy 1.24+ because np.int was removed
    n_jobs, n_per_job, starts = _partition_estimators(10, 3)
    assert n_jobs == 3
    assert n_per_job == [4, 3, 3]
    assert starts == [0, 4, 7, 10]


def test_check_random_state():
    assert isinstance(check_random_state(0), np.random.RandomState)
    random_state = np.random.RandomState(1)
    assert check_random_state(random_state) is random_state

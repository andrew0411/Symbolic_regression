import pickle

import numpy as np
import pytest
from sklearn.base import clone, is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import get_tags

from gp import (SymbolicClassifier, SymbolicRegressor, SymbolicTransformer,
                make_function)


@pytest.fixture(scope='module')
def toy():
    rng = np.random.RandomState(0)
    X = rng.uniform(-1, 1, (200, 2))
    y = X[:, 0] ** 2 - X[:, 1] + 0.5
    return X, y


def small(**params):
    defaults = dict(population_size=1000, generations=10,
                    function_set=('add', 'sub', 'mul', 'pow2'),
                    parsimony_coefficient=0.001, random_state=0)
    defaults.update(params)
    return SymbolicRegressor(**defaults)


def test_recovers_known_formula(toy):
    X, y = toy
    est = small().fit(X, y)
    assert est.score(X, y) > 0.99, str(est)


def test_same_seed_same_program(toy):
    X, y = toy
    assert str(small(generations=3).fit(X, y)) == \
        str(small(generations=3).fit(X, y))


def test_n_jobs_does_not_change_result(toy):
    X, y = toy
    serial = small(generations=3, n_jobs=1).fit(X, y)
    parallel = small(generations=3, n_jobs=2).fit(X, y)
    assert str(serial) == str(parallel)


def test_sklearn_integration(toy):
    X, y = toy
    est = small(generations=3)
    assert is_regressor(est)
    assert clone(est).get_params() == est.get_params()
    pipe = make_pipeline(StandardScaler(), small(generations=3))
    assert cross_val_score(pipe, X, y, cv=3).shape == (3,)


def test_pickle_round_trip(toy):
    X, y = toy
    est = small(generations=3).fit(X, y)
    restored = pickle.loads(pickle.dumps(est))
    np.testing.assert_allclose(restored.predict(X), est.predict(X))


def test_custom_function_and_input_checks(toy):
    X, y = toy
    exp = make_function(function=lambda x1: np.exp(np.clip(x1, -50., 50.)),
                        name='exp', arity=1, wrap=False)
    est = small(generations=2, function_set=('add', 'mul', exp)).fit(X, y)
    assert np.all(np.isfinite(est.predict(X)))
    with pytest.raises(ValueError):
        est.predict(X[:, :1])


def test_predict_before_fit_raises():
    with pytest.raises(NotFittedError):
        small().predict(np.zeros((2, 2)))


def test_classifier(toy):
    X, _ = toy
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = SymbolicClassifier(population_size=300, generations=3,
                             random_state=0).fit(X, y)
    assert is_classifier(clf)
    assert not get_tags(clf).classifier_tags.multi_class
    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.)
    assert clf.score(X, y) > 0.9


def test_transformer(toy):
    X, y = toy
    trans = SymbolicTransformer(population_size=300, generations=2,
                                hall_of_fame=20, n_components=5,
                                random_state=0)
    assert trans.fit_transform(X, y).shape == (len(X), 5)

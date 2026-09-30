import numpy as np
import pytest

from gp import SymbolicRegressor
from gp.fitness import _fitness_map
from gp.functions import _function_map as F
from gp.program import _Program
from gp.utils import check_random_state


def make_program(program, metric='rmse'):
    function_set = list(F.values())
    arities = {}
    for function in function_set:
        arities.setdefault(function.arity, []).append(function)
    return _Program(function_set=function_set, arities=arities,
                    init_depth=(2, 6), init_method='half and half',
                    n_features=2, const_range=(-1., 1.),
                    metric=_fitness_map[metric], p_point_replace=0.05,
                    parsimony_coefficient=0.01,
                    random_state=check_random_state(0), program=program)


def test_execute_hand_built_program():
    # add(mul(X0, X1), 0.5)
    program = make_program([F['add'], F['mul'], 0, 1, 0.5])
    X = np.array([[1., 2.], [3., 4.]])
    np.testing.assert_allclose(program.execute(X), [2.5, 12.5])
    assert program.length_ == 5
    assert program.depth_ == 2
    assert str(program) == 'add(mul(X0, X1), 0.500)'


def test_incomplete_program_is_rejected():
    with pytest.raises(ValueError):
        make_program([F['add'], 0])


@pytest.mark.parametrize('metric', ['rmse', 'mse', 'mean absolute error'])
def test_non_finite_fitness_is_worst(metric):
    # Nesting pow3 five times at X0=100 overflows to inf, and sin(inf) = NaN
    program = make_program([F['sin']] + [F['pow3']] * 5 + [0], metric)
    X = np.array([[100., 0.], [100., 0.]])
    assert program.raw_fitness(X, np.zeros(2), np.ones(2)) == np.inf


def test_auto_parsimony_survives_overflow():
    # With inputs around 1e100, a single nested pow3 or mul already overflows
    rng = np.random.RandomState(0)
    X = rng.uniform(1e99, 1e100, (100, 2))
    est = SymbolicRegressor(population_size=300, generations=1,
                            function_set=('add', 'mul', 'pow3', 'sin'),
                            parsimony_coefficient='auto', random_state=0)
    est.fit(X, X[:, 0])
    population = est._programs[-1]
    # Precondition: the test is only meaningful if some programs overflowed
    assert any(np.isinf(p.raw_fitness_) for p in population)
    assert not any(np.isnan(p.fitness_) for p in population)

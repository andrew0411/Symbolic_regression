import numpy as np
import pytest

from gp.functions import _function_map, make_function


@pytest.mark.parametrize('name', sorted(_function_map))
def test_builtin_functions_are_closed(name):
    # 0, 음수, 0 근처 입력에서도 유한값을 내야 한다 (gplearn closure 규약)
    function = _function_map[name]
    for value in (0., -1., 1e-4):
        args = [np.full(5, value) for _ in range(function.arity)]
        assert np.all(np.isfinite(function(*args)))


def test_pow_functions():
    x = np.array([-2., 0., 3.])
    np.testing.assert_allclose(_function_map['pow2'](x), [4., 0., 9.])
    np.testing.assert_allclose(_function_map['pow3'](x), [-8., 0., 27.])


def test_make_function_rejects_wrong_arity():
    with pytest.raises(ValueError):
        make_function(function=lambda x1, x2: x1 + x2, name='bad', arity=1)

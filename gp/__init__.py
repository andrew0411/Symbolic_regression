"""Genetic Programming symbolic regression, adapted from gplearn.

Original: https://github.com/trevorstephens/gplearn (BSD 3 clause, see gp/LICENSE)
"""

from .fitness import make_fitness
from .functions import make_function
from .genetics import SymbolicClassifier, SymbolicRegressor, SymbolicTransformer

__version__ = '0.1.0'

__all__ = ['SymbolicRegressor', 'SymbolicClassifier', 'SymbolicTransformer',
           'make_function', 'make_fitness']

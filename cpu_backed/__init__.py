"""
CPU Backend - Optimized CPU implementations for HH simulations.
"""

from .vectorized import (
    VectorizedSimulator,
    OptimizedRK4Integrator,
    BatchSimulator
)

# Try to import Numba kernels (optional dependency)
try:
    from .numba_kernels import NumbaSimulator
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    NumbaSimulator = None

__all__ = [
    'VectorizedSimulator',
    'OptimizedRK4Integrator',
    'BatchSimulator',
    'NumbaSimulator',
    'NUMBA_AVAILABLE'
]

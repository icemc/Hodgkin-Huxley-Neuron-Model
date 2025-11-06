"""
CPU Backend - Optimized CPU implementations for HH simulations.
"""

from .vectorized import (
    VectorizedSimulator,
    OptimizedRK4Integrator,
    BatchSimulator
)

__all__ = [
    'VectorizedSimulator',
    'OptimizedRK4Integrator',
    'BatchSimulator'
]

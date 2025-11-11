"""
GPU-backed implementation using CuPy for batch neuron simulations.

This module provides GPU-accelerated Hodgkin-Huxley simulations
optimized for large batch sizes (spatial parallelism across neurons).
"""

from .gpu_simulator import GPUSimulator

__all__ = ['GPUSimulator']

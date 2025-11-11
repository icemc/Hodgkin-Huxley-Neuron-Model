"""
Unified Simulator Interface for Hodgkin-Huxley Model.

This module provides a single entry point for simulating HH neurons
with automatic backend selection (CPU or GPU).
"""

import numpy as np
from typing import Optional, List
import warnings

from hh_core.api import HHModel, Stimulus, SimulationResult, BaseSimulator
from hh_core.models import HHState


def Simulator(backend: str = 'cpu',
              model: Optional[HHModel] = None,
              integrator: str = 'rk4',
              dtype=np.float64) -> BaseSimulator:
    """
    Create a Hodgkin-Huxley simulator with the specified backend.
    
    This is a factory function that returns the appropriate simulator
    implementation based on the backend selection.
    
    Args:
        backend: 'cpu' or 'gpu'
        model: HH model (creates default if None)
        integrator: Integration method
            - CPU: 'euler', 'rk4', 'rk4rl', 'rk4-scipy'
            - GPU: only 'rk4' (parameter ignored)
        dtype: Data type for arrays
            - CPU: np.float32 or np.float64
            - GPU: only float32 supported
    
    Returns:
        Simulator instance (CPUSimulator or GPUSimulator)
    
    Examples:
        # CPU simulation
        >>> sim = Simulator(backend='cpu', integrator='rk4')
        >>> result = sim.run(T=100.0, dt=0.01, batch_size=1)
        
        # GPU simulation
        >>> sim = Simulator(backend='gpu', dtype=np.float32)
        >>> result = sim.run(T=100.0, dt=0.01, batch_size=10000)
    """
    backend = backend.lower()
    
    if backend == 'cpu':
        from cpu_backed import CPUSimulator
        return CPUSimulator(model=model, integrator=integrator, dtype=dtype)
    
    elif backend == 'gpu':
        # Check if CuPy is available
        try:
            from gpu_backed import GPUSimulator
        except ImportError:
            raise ImportError(
                "GPU backend requires CuPy. Install with:\n"
                "  pip install cupy-cuda11x  (for CUDA 11.x)\n"
                "  pip install cupy-cuda12x  (for CUDA 12.x)"
            )
        
        # GPU only supports float32
        if dtype != np.float32:
            warnings.warn(
                f"GPU backend only supports float32, ignoring dtype={dtype}",
                UserWarning
            )
        
        return GPUSimulator(model=model, integrator=integrator, dtype='float32')
    
    else:
        raise ValueError(
            f"Unknown backend: '{backend}'. "
            f"Valid options are 'cpu' or 'gpu'."
        )


# Re-export core classes for convenience
__all__ = [
    'Simulator',
    'HHModel',
    'Stimulus',
    'SimulationResult',
]

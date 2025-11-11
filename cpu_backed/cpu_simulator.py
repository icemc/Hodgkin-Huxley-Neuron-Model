"""
CPU-backed simulator for Hodgkin-Huxley model.

This module provides a high-level Simulator class that uses
the vectorized CPU backend for efficient simulations.
"""

import numpy as np
from typing import Optional, List
import warnings

from hh_core.api import HHModel, Stimulus, SimulationResult, BaseSimulator
from hh_core.models import HHState
from .vectorized import VectorizedSimulator


class CPUSimulator(BaseSimulator):
    """
    CPU-based simulator for HH neurons.
    
    Uses vectorized NumPy implementation optimized for batch simulations.
    """
    
    def __init__(self, 
                 model: Optional[HHModel] = None,
                 integrator: str = 'rk4',
                 dtype=np.float64):
        """
        Initialize CPU simulator.
        
        Args:
            model: HH model (creates default if None)
            integrator: 'euler', 'rk4', 'rk4rl' (RK4 with Rush-Larsen), or 'rk4-scipy'
            dtype: Data type for arrays (np.float32 or np.float64)
        """
        super().__init__(model, integrator, dtype)
        
        # Create backend simulator
        self.sim = VectorizedSimulator(
            self.model.params, integrator, dtype
        )
    
    def run(self,
            T: float,
            dt: float = 0.01,
            state0: Optional[HHState] = None,
            stimulus: Optional[np.ndarray] = None,
            batch_size: int = 1,
            record: Optional[List[str]] = None,
            spike_threshold: float = 0.0) -> SimulationResult:
        """
        Run simulation.
        
        Args:
            T: Total simulation time (ms)
            dt: Time step (ms) - default 0.01 ms is typical for HH
            state0: Initial state (creates resting state if None)
            stimulus: External current array
            batch_size: Number of neurons (ignored if state0 provided)
            record: Variables to record ['V', 'm', 'h', 'n'] (default: all)
            spike_threshold: Threshold for spike detection (mV)
        
        Returns:
            SimulationResult object with recorded data
        """
        # Validate dt
        if dt > 0.1:
            warnings.warn(f"Large dt ({dt} ms) may cause numerical instability. "
                        f"Recommended: dt <= 0.05 ms for HH model.")
        
        # Setup initial state
        if state0 is None:
            state0 = self.model.resting_state(batch_size)
        
        # Default recording
        if record is None:
            record = ['V', 'm', 'h', 'n']
        
        # Run simulation
        results = self.sim.run(
            T=T, dt=dt, state0=state0, stimulus=stimulus,
            batch_size=batch_size, record_vars=record,
            spike_threshold=spike_threshold
        )
        
        return SimulationResult(results, self.model.params, dt)


# Re-export for convenience
__all__ = ['CPUSimulator', 'HHModel', 'Stimulus', 'SimulationResult']

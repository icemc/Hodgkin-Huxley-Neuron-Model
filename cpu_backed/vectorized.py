"""
Vectorized NumPy implementation of HH simulation for CPU.

This module provides highly optimized vectorized implementations
using NumPy for efficient batch processing on CPUs.
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hh_core.models import (
    HHState, HHParameters, derivatives, 
    compute_currents, rush_larsen_gating_step
)
from hh_core.integrators import IntegratorBase, RK4, RK4RushLarsen, ForwardEuler, RK45Scipy
from hh_core.utils import detect_spikes, interpolate_spike_times


class VectorizedSimulator:
    """
    Vectorized CPU-based simulator for Hodgkin-Huxley neurons.
    
    Optimized for large batch sizes using NumPy vectorization.
    """
    
    def __init__(self, params: Optional[HHParameters] = None, 
                 integrator_type: str = 'rk4',
                 dtype=np.float64):
        """
        Initialize vectorized simulator.
        
        Args:
            params: Model parameters (uses defaults if None)
            integrator_type: 'euler', 'rk4', 'rk4rl' (RK4 with Rush-Larsen), or 'rk4-scipy'
            dtype: Data type for arrays (float32 or float64)
        """
        self.params = params if params is not None else HHParameters()
        self.dtype = dtype
        self.integrator_type = integrator_type
        
        # Create integrator
        if integrator_type == 'euler':
            self.integrator = ForwardEuler(self.params)
        elif integrator_type == 'rk4':
            self.integrator = RK4(self.params)
        elif integrator_type == 'rk4rl':
            self.integrator = RK4RushLarsen(self.params)
        elif integrator_type == 'rk4-scipy':
            self.integrator = RK45Scipy(self.params)
        else:
            raise ValueError(f"Unknown integrator type: {integrator_type}")
    
    def run(self, 
            T: float,
            dt: float,
            state0: Optional[HHState] = None,
            stimulus: Optional[np.ndarray] = None,
            batch_size: int = 1,
            record_vars: Optional[List[str]] = None,
            spike_threshold: float = 0.0,
            detect_spikes_flag: bool = True) -> Dict:
        """
        Run simulation.
        
        Args:
            T: Total simulation time (ms)
            dt: Time step (ms)
            state0: Initial state (creates resting state if None)
            stimulus: External current array (n_steps,) or (n_steps, batch_size)
            batch_size: Number of neurons (ignored if state0 provided)
            record_vars: Variables to record ['V', 'm', 'h', 'n']
            spike_threshold: Threshold for spike detection (mV)
            detect_spikes_flag: Whether to detect spikes
        
        Returns:
            Dictionary with simulation results
        """
        # Initialize state
        if state0 is None:
            state0 = HHState.resting_state(batch_size, dtype=self.dtype)
        else:
            # Ensure correct dtype
            if state0.data.dtype != self.dtype:
                state0.data = state0.data.astype(self.dtype)
        
        # Determine batch size
        if state0.data.ndim == 1:
            actual_batch_size = 1
        else:
            actual_batch_size = state0.data.shape[0]
        
        # Setup time
        n_steps = int(np.ceil(T / dt))
        time = np.linspace(0, T, n_steps + 1)
        
        # Setup stimulus
        if stimulus is None:
            stimulus = np.zeros(n_steps, dtype=self.dtype)
        else:
            stimulus = np.asarray(stimulus, dtype=self.dtype)
        
        # Ensure stimulus has correct shape
        if stimulus.ndim == 1:
            if actual_batch_size > 1:
                # Broadcast to all neurons
                stimulus_array = np.tile(stimulus[:, np.newaxis], (1, actual_batch_size))
            else:
                stimulus_array = stimulus
        else:
            stimulus_array = stimulus
        
        # Determine what to record
        if record_vars is None:
            record_vars = ['V', 'm', 'h', 'n']
        
        # Initialize recording arrays
        results = self._initialize_results(
            n_steps, actual_batch_size, record_vars, time
        )
        
        # Record initial state
        self._record_state(results, 0, state0, record_vars)
        
        # Main simulation loop
        state = state0
        for i in range(n_steps):
            # Get stimulus for this step
            if stimulus_array.ndim == 1:
                I_ext = stimulus_array[i]
            else:
                I_ext = stimulus_array[i, :]
            
            # Integration step
            state = self.integrator.step(state, dt, I_ext)
            
            # Record state
            self._record_state(results, i + 1, state, record_vars)
        
        # Detect spikes
        if detect_spikes_flag and 'V' in record_vars:
            results['spikes'] = self._detect_spikes_batch(
                results['V'], time, spike_threshold, actual_batch_size
            )
        
        results['state'] = state
        return results
    
    def _initialize_results(self, n_steps: int, batch_size: int, 
                          record_vars: List[str], time: np.ndarray) -> Dict:
        """Initialize result arrays."""
        results = {'time': time}
        
        for var in record_vars:
            if batch_size == 1:
                results[var] = np.zeros(n_steps + 1, dtype=self.dtype)
            else:
                results[var] = np.zeros((n_steps + 1, batch_size), dtype=self.dtype)
        
        return results
    
    def _record_state(self, results: Dict, idx: int, state: HHState, 
                     record_vars: List[str]):
        """Record state at given index."""
        if 'V' in record_vars:
            results['V'][idx] = state.V
        if 'm' in record_vars:
            results['m'][idx] = state.m
        if 'h' in record_vars:
            results['h'][idx] = state.h
        if 'n' in record_vars:
            results['n'][idx] = state.n
    
    def _detect_spikes_batch(self, voltage: np.ndarray, time: np.ndarray,
                            threshold: float, batch_size: int) -> Dict:
        """Detect spikes for batch of neurons."""
        if batch_size == 1:
            spike_indices = detect_spikes(voltage, threshold)
            spike_times = interpolate_spike_times(voltage, time, spike_indices, threshold)
            return {
                'indices': spike_indices,
                'times': spike_times,
                'count': len(spike_indices)
            }
        else:
            spike_data = []
            for i in range(batch_size):
                v = voltage[:, i]
                spike_indices = detect_spikes(v, threshold)
                spike_times = interpolate_spike_times(v, time, spike_indices, threshold)
                spike_data.append({
                    'indices': spike_indices,
                    'times': spike_times,
                    'count': len(spike_indices)
                })
            return spike_data


class OptimizedRK4Integrator:
    """
    Memory-optimized RK4 integrator with pre-allocated temporaries.
    
    Reduces allocation overhead by reusing arrays across steps.
    """
    
    def __init__(self, params: HHParameters, batch_size: int, dtype=np.float64):
        """
        Initialize with pre-allocated arrays.
        
        Args:
            params: Model parameters
            batch_size: Number of neurons in batch
            dtype: Data type for arrays
        """
        self.params = params
        self.batch_size = batch_size
        self.dtype = dtype
        
        # Pre-allocate temporary arrays
        shape = (batch_size, 4) if batch_size > 1 else (4,)
        self.k1 = np.zeros(shape, dtype=dtype)
        self.k2 = np.zeros(shape, dtype=dtype)
        self.k3 = np.zeros(shape, dtype=dtype)
        self.k4 = np.zeros(shape, dtype=dtype)
        self.temp_state_data = np.zeros(shape, dtype=dtype)
    
    def step(self, state: HHState, dt: float, I_ext: np.ndarray) -> HHState:
        """RK4 step with pre-allocated arrays."""
        # k1
        self.k1[:] = derivatives(state, I_ext, self.params)
        
        # k2
        self.temp_state_data[:] = state.data + 0.5 * dt * self.k1
        temp_state = HHState(self.temp_state_data)
        self.k2[:] = derivatives(temp_state, I_ext, self.params)
        
        # k3
        self.temp_state_data[:] = state.data + 0.5 * dt * self.k2
        temp_state = HHState(self.temp_state_data)
        self.k3[:] = derivatives(temp_state, I_ext, self.params)
        
        # k4
        self.temp_state_data[:] = state.data + dt * self.k3
        temp_state = HHState(self.temp_state_data)
        self.k4[:] = derivatives(temp_state, I_ext, self.params)
        
        # Combine
        new_data = state.data + (dt / 6.0) * (self.k1 + 2*self.k2 + 2*self.k3 + self.k4)
        return HHState(new_data)


class BatchSimulator:
    """
    Simulator optimized for large batches with parameter variations.
    
    Supports different parameters for each neuron in the batch.
    """
    
    def __init__(self, dtype=np.float64):
        """Initialize batch simulator."""
        self.dtype = dtype
    
    def run_parameter_sweep(self,
                          param_values: Dict[str, np.ndarray],
                          T: float,
                          dt: float,
                          stimulus: np.ndarray,
                          integrator_type: str = 'rk4',
                          record_vars: Optional[List[str]] = None,
                          spike_threshold: float = 0.0) -> List[Dict]:
        """
        Run simulation with parameter sweep.
        
        Args:
            param_values: Dictionary mapping parameter names to arrays of values
            T: Simulation time (ms)
            dt: Time step (ms)
            stimulus: Stimulus array
            integrator_type: Type of integrator
            record_vars: Variables to record
            spike_threshold: Spike detection threshold
        
        Returns:
            List of result dictionaries, one per parameter combination
        """
        # For simplicity, we'll run each parameter set independently
        # A more sophisticated implementation could vectorize over parameters
        
        # Get all parameter combinations
        param_names = list(param_values.keys())
        n_combinations = len(param_values[param_names[0]])
        
        results = []
        for i in range(n_combinations):
            # Create parameter set
            params = HHParameters()
            for name in param_names:
                setattr(params, name, param_values[name][i])
            
            # Create simulator
            sim = VectorizedSimulator(params, integrator_type, self.dtype)
            
            # Run simulation
            result = sim.run(
                T=T, dt=dt, stimulus=stimulus,
                batch_size=1, record_vars=record_vars,
                spike_threshold=spike_threshold
            )
            
            # Store parameter values with result
            result['params'] = {name: param_values[name][i] for name in param_names}
            results.append(result)
        
        return results

"""
High-level API for Hodgkin-Huxley simulations.

This module provides user-friendly classes for running HH simulations
with CPU-optimized vectorized backend.
"""

import numpy as np
from typing import Optional, Union, List, Dict
import warnings

from hh_core.models import HHParameters, HHState
from hh_core.utils import Stimulus as StimGen
from cpu_backed.vectorized import VectorizedSimulator


class HHModel:
    """
    Hodgkin-Huxley neuron model.
    
    Encapsulates model parameters and provides a clean interface.
    """
    
    def __init__(self, params: Optional[HHParameters] = None):
        """
        Initialize HH model.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params if params is not None else HHParameters()
    
    def get_params(self) -> HHParameters:
        """Get model parameters."""
        return self.params
    
    def set_params(self, **kwargs):
        """
        Update model parameters.
        
        Example:
            model.set_params(g_Na=100.0, E_K=-80.0)
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise ValueError(f"Unknown parameter: {key}")
    
    def resting_state(self, batch_size: int = 1) -> HHState:
        """
        Get resting state for the model.
        
        Args:
            batch_size: Number of neurons
        
        Returns:
            Resting state
        """
        return HHState.resting_state(batch_size)


class Stimulus:
    """
    Stimulus generation wrapper.
    
    Provides convenient methods for creating common stimulus patterns.
    """
    
    @staticmethod
    def constant(amplitude: float, duration: float, dt: float) -> np.ndarray:
        """Generate constant current."""
        return StimGen.constant(amplitude, duration, dt)
    
    @staticmethod
    def step(amplitude: float, t_start: float, t_end: float,
            duration: float, dt: float) -> np.ndarray:
        """Generate step current."""
        return StimGen.step(amplitude, t_start, t_end, duration, dt)
    
    @staticmethod
    def pulse_train(amplitude: float, pulse_duration: float,
                   pulse_period: float, n_pulses: int,
                   t_start: float, duration: float, dt: float) -> np.ndarray:
        """Generate pulse train."""
        return StimGen.pulse_train(amplitude, pulse_duration, pulse_period,
                                   n_pulses, t_start, duration, dt)
    
    @staticmethod
    def noisy(mean: float, std: float, duration: float, dt: float,
             seed: Optional[int] = None) -> np.ndarray:
        """Generate noisy current."""
        return StimGen.noisy(mean, std, duration, dt, seed)
    
    @staticmethod
    def ramp(start_amplitude: float, end_amplitude: float,
            duration: float, dt: float) -> np.ndarray:
        """Generate ramping current."""
        return StimGen.ramp(start_amplitude, end_amplitude, duration, dt)


class Simulator:
    """
    Main simulator class for HH neurons.
    
    Uses vectorized NumPy implementation optimized for batch simulations.
    """
    
    def __init__(self, 
                 model: Optional[HHModel] = None,
                 backend: str = 'cpu',
                 integrator: str = 'rk4',
                 dtype=np.float64):
        """
        Initialize simulator.
        
        Args:
            model: HH model (creates default if None)
            backend: 'cpu' (only supported backend)
            integrator: 'euler', 'rk4', or 'rk4rl' (RK4 with Rush-Larsen)
            dtype: Data type for arrays (np.float32 or np.float64)
        """
        self.model = model if model is not None else HHModel()
        self.backend = backend
        self.integrator = integrator
        self.dtype = dtype
        
        # Create backend simulator
        if backend == 'cpu':
            self.sim = VectorizedSimulator(
                self.model.params, integrator, dtype
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Only 'cpu' backend is supported.")
    
    def run(self,
            T: float,
            dt: float = 0.01,
            state0: Optional[HHState] = None,
            stimulus: Optional[np.ndarray] = None,
            batch_size: int = 1,
            record: Optional[List[str]] = None,
            spike_threshold: float = 0.0) -> 'SimulationResult':
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
        
        # Run simulation based on backend
        if self.backend == 'cpu':
            results = self.sim.run(
                T=T, dt=dt, state0=state0, stimulus=stimulus,
                batch_size=batch_size, record_vars=record,
                spike_threshold=spike_threshold
            )
        
        return SimulationResult(results, self.model.params, dt)


class SimulationResult:
    """
    Container for simulation results.
    
    Provides convenient access to recorded data and analysis methods.
    """
    
    def __init__(self, data: Dict, params: HHParameters, dt: float):
        """
        Initialize result container.
        
        Args:
            data: Dictionary with simulation data
            params: Model parameters used
            dt: Time step used
        """
        self.data = data
        self.params = params
        self.dt = dt
    
    @property
    def time(self) -> np.ndarray:
        """Time array."""
        return self.data['time']
    
    @property
    def V(self) -> np.ndarray:
        """Voltage trace."""
        return self.data.get('V', None)
    
    @property
    def m(self) -> np.ndarray:
        """Sodium activation gating variable."""
        return self.data.get('m', None)
    
    @property
    def h(self) -> np.ndarray:
        """Sodium inactivation gating variable."""
        return self.data.get('h', None)
    
    @property
    def n(self) -> np.ndarray:
        """Potassium activation gating variable."""
        return self.data.get('n', None)
    
    @property
    def spikes(self):
        """Spike detection results."""
        return self.data.get('spikes', None)
    
    def get_spike_count(self, neuron_idx: Optional[int] = None) -> Union[int, List[int]]:
        """
        Get spike count.
        
        Args:
            neuron_idx: Neuron index (None for single neuron or all neurons)
        
        Returns:
            Spike count(s)
        """
        if self.spikes is None:
            return 0
        
        if isinstance(self.spikes, dict):
            # Single neuron
            return self.spikes['count']
        else:
            # Batch
            if neuron_idx is not None:
                return self.spikes[neuron_idx]['count']
            else:
                return [s['count'] for s in self.spikes]
    
    def get_spike_times(self, neuron_idx: Optional[int] = None):
        """
        Get spike times.
        
        Args:
            neuron_idx: Neuron index (None for single neuron)
        
        Returns:
            Spike times array
        """
        if self.spikes is None:
            return np.array([])
        
        if isinstance(self.spikes, dict):
            # Single neuron
            return self.spikes['times']
        else:
            # Batch
            if neuron_idx is None:
                return [s['times'] for s in self.spikes]
            else:
                return self.spikes[neuron_idx]['times']
    
    def plot(self, variables: Optional[List[str]] = None, 
            neuron_idx: Optional[int] = None,
            figsize=(12, 8)):
        """
        Plot simulation results.
        
        Args:
            variables: Variables to plot (default: all recorded)
            neuron_idx: Which neuron to plot for batch simulations
            figsize: Figure size
        
        Returns:
            matplotlib figure and axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plotting. Install with: pip install matplotlib")
        
        if variables is None:
            variables = []
            if self.V is not None: variables.append('V')
            if self.m is not None: variables.append('m')
            if self.h is not None: variables.append('h')
            if self.n is not None: variables.append('n')
        
        n_plots = len(variables)
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, sharex=True)
        
        if n_plots == 1:
            axes = [axes]
        
        # Determine if batch or single
        is_batch = self.V is not None and self.V.ndim == 2
        
        for i, var in enumerate(variables):
            ax = axes[i]
            
            data_var = getattr(self, var)
            if data_var is None:
                continue
            
            if is_batch:
                if neuron_idx is None:
                    # Plot first neuron
                    idx = 0
                else:
                    idx = neuron_idx
                ax.plot(self.time, data_var[:, idx])
            else:
                ax.plot(self.time, data_var)
            
            ax.set_ylabel(var)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (ms)')
        fig.suptitle('Hodgkin-Huxley Simulation Results')
        plt.tight_layout()
        
        return fig, axes
    
    def summary(self) -> str:
        """
        Get text summary of simulation results.
        
        Returns:
            Summary string
        """
        lines = ["Simulation Results Summary"]
        lines.append("=" * 40)
        lines.append(f"Duration: {self.time[-1]:.2f} ms")
        lines.append(f"Time step: {self.dt:.4f} ms")
        lines.append(f"Number of steps: {len(self.time)}")
        
        if self.V is not None:
            if self.V.ndim == 1:
                lines.append(f"Neurons: 1")
                lines.append(f"V range: [{self.V.min():.2f}, {self.V.max():.2f}] mV")
            else:
                lines.append(f"Neurons: {self.V.shape[1]}")
                lines.append(f"V range: [{self.V.min():.2f}, {self.V.max():.2f}] mV")
        
        if self.spikes is not None:
            if isinstance(self.spikes, dict):
                count = self.spikes['count']
                lines.append(f"Spikes: {count}")
                if count > 0:
                    rate = count / (self.time[-1] / 1000.0)
                    lines.append(f"Firing rate: {rate:.2f} Hz")
            else:
                counts = [s['count'] for s in self.spikes]
                lines.append(f"Total spikes: {sum(counts)}")
                lines.append(f"Mean spikes per neuron: {np.mean(counts):.2f}")
        
        return "\n".join(lines)

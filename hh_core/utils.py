"""
Utility functions for stimulus generation, spike detection, and analysis.
"""

import numpy as np
from typing import Optional, Tuple, List


class Stimulus:
    """
    Stimulus generator for neuron simulations.
    
    Provides various types of current injection patterns.
    """
    
    @staticmethod
    def constant(amplitude: float, duration: float, dt: float) -> np.ndarray:
        """
        Generate constant current injection.
        
        Args:
            amplitude: Current amplitude (uA/cm^2)
            duration: Total duration (ms)
            dt: Time step (ms)
        
        Returns:
            Array of current values
        """
        n_steps = int(np.ceil(duration / dt))
        return np.full(n_steps, amplitude)
    
    @staticmethod
    def step(amplitude: float, t_start: float, t_end: float, 
             duration: float, dt: float) -> np.ndarray:
        """
        Generate step current (zero, then amplitude, then zero).
        
        Args:
            amplitude: Current amplitude during step (uA/cm^2)
            t_start: Time when step starts (ms)
            t_end: Time when step ends (ms)
            duration: Total duration (ms)
            dt: Time step (ms)
        
        Returns:
            Array of current values
        """
        n_steps = int(np.ceil(duration / dt))
        time = np.linspace(0, duration, n_steps)
        current = np.zeros(n_steps)
        mask = (time >= t_start) & (time < t_end)
        current[mask] = amplitude
        return current
    
    @staticmethod
    def pulse_train(amplitude: float, pulse_duration: float, 
                   pulse_period: float, n_pulses: int,
                   t_start: float, duration: float, dt: float) -> np.ndarray:
        """
        Generate train of current pulses.
        
        Args:
            amplitude: Pulse amplitude (uA/cm^2)
            pulse_duration: Duration of each pulse (ms)
            pulse_period: Period between pulse starts (ms)
            n_pulses: Number of pulses
            t_start: Time of first pulse (ms)
            duration: Total duration (ms)
            dt: Time step (ms)
        
        Returns:
            Array of current values
        """
        n_steps = int(np.ceil(duration / dt))
        time = np.linspace(0, duration, n_steps)
        current = np.zeros(n_steps)
        
        for i in range(n_pulses):
            pulse_start = t_start + i * pulse_period
            pulse_end = pulse_start + pulse_duration
            mask = (time >= pulse_start) & (time < pulse_end)
            current[mask] = amplitude
        
        return current
    
    @staticmethod
    def noisy(mean: float, std: float, duration: float, dt: float,
             seed: Optional[int] = None) -> np.ndarray:
        """
        Generate noisy (Gaussian white noise) current.
        
        Args:
            mean: Mean current (uA/cm^2)
            std: Standard deviation (uA/cm^2)
            duration: Total duration (ms)
            dt: Time step (ms)
            seed: Random seed for reproducibility
        
        Returns:
            Array of current values
        """
        if seed is not None:
            np.random.seed(seed)
        
        n_steps = int(np.ceil(duration / dt))
        current = np.random.normal(mean, std, n_steps)
        return current
    
    @staticmethod
    def ramp(start_amplitude: float, end_amplitude: float,
            duration: float, dt: float) -> np.ndarray:
        """
        Generate linearly ramping current.
        
        Args:
            start_amplitude: Initial current (uA/cm^2)
            end_amplitude: Final current (uA/cm^2)
            duration: Total duration (ms)
            dt: Time step (ms)
        
        Returns:
            Array of current values
        """
        n_steps = int(np.ceil(duration / dt))
        current = np.linspace(start_amplitude, end_amplitude, n_steps)
        return current


def detect_spikes(voltage: np.ndarray, threshold: float = 0.0,
                 min_interval: Optional[int] = None) -> np.ndarray:
    """
    Detect spike times using threshold crossing.
    
    Detects upward crossings of the threshold (V[i] >= threshold and V[i-1] < threshold).
    
    Args:
        voltage: Array of voltage values (1D for single neuron, 2D for batch)
        threshold: Spike detection threshold (mV)
        min_interval: Minimum samples between spikes (refractory period)
    
    Returns:
        For 1D input: array of spike indices
        For 2D input: list of arrays, one per neuron
    """
    if voltage.ndim == 1:
        # Single neuron
        crossings = (voltage[1:] >= threshold) & (voltage[:-1] < threshold)
        spike_indices = np.where(crossings)[0] + 1
        
        # Apply minimum interval if specified
        if min_interval is not None and len(spike_indices) > 0:
            filtered_spikes = [spike_indices[0]]
            for spike_idx in spike_indices[1:]:
                if spike_idx - filtered_spikes[-1] >= min_interval:
                    filtered_spikes.append(spike_idx)
            spike_indices = np.array(filtered_spikes)
        
        return spike_indices
    
    else:
        # Batch of neurons
        n_neurons = voltage.shape[1] if voltage.ndim == 2 else voltage.shape[0]
        all_spikes = []
        
        for i in range(n_neurons):
            if voltage.ndim == 2:
                v = voltage[:, i]
            else:
                v = voltage[i, :]
            spikes = detect_spikes(v, threshold, min_interval)
            all_spikes.append(spikes)
        
        return all_spikes


def interpolate_spike_times(voltage: np.ndarray, time: np.ndarray,
                           spike_indices: np.ndarray,
                           threshold: float = 0.0) -> np.ndarray:
    """
    Interpolate precise spike times using linear interpolation.
    
    Args:
        voltage: Voltage trace
        time: Time array
        spike_indices: Indices where spikes were detected
        threshold: Threshold value
    
    Returns:
        Array of interpolated spike times
    """
    if len(spike_indices) == 0:
        return np.array([])
    
    spike_times = np.zeros(len(spike_indices))
    
    for i, idx in enumerate(spike_indices):
        if idx == 0:
            spike_times[i] = time[idx]
        else:
            # Linear interpolation between idx-1 and idx
            v0 = voltage[idx - 1]
            v1 = voltage[idx]
            t0 = time[idx - 1]
            t1 = time[idx]
            
            # t_spike = t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
            if abs(v1 - v0) > 1e-10:
                t_spike = t0 + (threshold - v0) / (v1 - v0) * (t1 - t0)
            else:
                t_spike = t0
            
            spike_times[i] = t_spike
    
    return spike_times


def compute_spike_statistics(spike_times: np.ndarray) -> dict:
    """
    Compute basic spike train statistics.
    
    Args:
        spike_times: Array of spike times (ms)
    
    Returns:
        Dictionary with:
            - 'count': number of spikes
            - 'rate': mean firing rate (Hz), requires duration info
            - 'isi_mean': mean inter-spike interval (ms)
            - 'isi_std': standard deviation of ISI (ms)
            - 'isi_cv': coefficient of variation of ISI
    """
    n_spikes = len(spike_times)
    
    stats = {'count': n_spikes}
    
    if n_spikes < 2:
        stats['isi_mean'] = np.nan
        stats['isi_std'] = np.nan
        stats['isi_cv'] = np.nan
    else:
        isis = np.diff(spike_times)
        stats['isi_mean'] = np.mean(isis)
        stats['isi_std'] = np.std(isis)
        stats['isi_cv'] = stats['isi_std'] / stats['isi_mean'] if stats['isi_mean'] > 0 else np.nan
    
    return stats


def compute_firing_rate(spike_times: np.ndarray, duration: float) -> float:
    """
    Compute mean firing rate.
    
    Args:
        spike_times: Array of spike times (ms)
        duration: Total duration of recording (ms)
    
    Returns:
        Mean firing rate (Hz)
    """
    if duration <= 0:
        return 0.0
    
    # Convert ms to seconds for Hz
    return len(spike_times) / (duration / 1000.0)


def create_batched_stimulus(base_stimulus: np.ndarray, 
                            batch_size: int,
                            variation: Optional[str] = None,
                            variation_params: Optional[dict] = None) -> np.ndarray:
    """
    Create batched stimulus from a base pattern.
    
    Args:
        base_stimulus: Base stimulus array (1D)
        batch_size: Number of neurons in batch
        variation: Type of variation ('scale', 'offset', 'noise', None)
        variation_params: Parameters for variation
    
    Returns:
        Batched stimulus array (n_steps, batch_size)
    """
    n_steps = len(base_stimulus)
    batched = np.tile(base_stimulus[:, np.newaxis], (1, batch_size))
    
    if variation is None:
        return batched
    
    if variation_params is None:
        variation_params = {}
    
    if variation == 'scale':
        # Scale each neuron's stimulus by a random factor
        scale_mean = variation_params.get('mean', 1.0)
        scale_std = variation_params.get('std', 0.1)
        scales = np.random.normal(scale_mean, scale_std, batch_size)
        batched = batched * scales[np.newaxis, :]
    
    elif variation == 'offset':
        # Add random offset to each neuron
        offset_mean = variation_params.get('mean', 0.0)
        offset_std = variation_params.get('std', 1.0)
        offsets = np.random.normal(offset_mean, offset_std, batch_size)
        batched = batched + offsets[np.newaxis, :]
    
    elif variation == 'noise':
        # Add independent noise to each neuron
        noise_std = variation_params.get('std', 0.1)
        noise = np.random.normal(0, noise_std, (n_steps, batch_size))
        batched = batched + noise
    
    return batched

"""
Example: Using the unified Simulator interface

This script demonstrates how to use the unified Simulator
to easily switch between CPU and GPU backends.
"""

import numpy as np
from simulator import Simulator, HHModel, Stimulus

def main():
    print("=" * 70)
    print("Unified Simulator Example")
    print("=" * 70)
    print()
    
    # Simulation parameters
    T = 50.0  # ms
    dt = 0.01  # ms
    
    # Create stimulus
    stimulus = Stimulus.step(
        amplitude=10.0,
        t_start=10.0,
        t_end=40.0,
        duration=T,
        dt=dt
    )
    
    # Example 1: CPU Simulator
    print("1. CPU Backend (RK4 integrator)")
    print("-" * 70)
    sim_cpu = Simulator(
        backend='cpu',
        integrator='rk4',
        dtype=np.float64
    )
    result_cpu = sim_cpu.run(T, dt, stimulus=stimulus, batch_size=1)
    print(f"   Spikes detected: {result_cpu.get_spike_count()}")
    print(f"   Voltage range: [{result_cpu.V.min():.2f}, {result_cpu.V.max():.2f}] mV")
    print()
    
    # Example 2: CPU Simulator with different integrator
    print("2. CPU Backend (RK4-Scipy integrator)")
    print("-" * 70)
    sim_scipy = Simulator(
        backend='cpu',
        integrator='rk4-scipy',
        dtype=np.float64
    )
    result_scipy = sim_scipy.run(T, dt, stimulus=stimulus, batch_size=1)
    print(f"   Spikes detected: {result_scipy.get_spike_count()}")
    print(f"   Voltage range: [{result_scipy.V.min():.2f}, {result_scipy.V.max():.2f}] mV")
    print()
    
    # Example 3: GPU Simulator
    print("3. GPU Backend")
    print("-" * 70)
    try:
        sim_gpu = Simulator(
            backend='gpu',
            dtype=np.float32  # GPU uses float32
        )
        result_gpu = sim_gpu.run(T, dt, stimulus=stimulus, batch_size=1)
        print(f"   Spikes detected: {result_gpu.get_spike_count()}")
        print(f"   Voltage range: [{result_gpu.V.min():.2f}, {result_gpu.V.max():.2f}] mV")
        print()
        
        # Example 4: GPU with large batch
        print("4. GPU Backend - Batch Simulation (1000 neurons)")
        print("-" * 70)
        result_batch = sim_gpu.run(T, dt, stimulus=stimulus, batch_size=1000)
        spike_counts = result_batch.get_spike_count()
        print(f"   Simulated: {result_batch.batch_size} neurons")
        print(f"   Total spikes: {sum(spike_counts)}")
        print(f"   Mean spikes per neuron: {np.mean(spike_counts):.2f}")
        print()
        
    except ImportError as e:
        print(f"   GPU not available: {e}")
        print("   Install CuPy to use GPU acceleration:")
        print("     pip install cupy-cuda11x  (for CUDA 11.x)")
        print("     pip install cupy-cuda12x  (for CUDA 12.x)")
        print()
    
    print("=" * 70)
    print("Example complete!")
    print("=" * 70)

if __name__ == '__main__':
    main()

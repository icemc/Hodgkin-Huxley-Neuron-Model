"""
Demo: GPU-accelerated Hodgkin-Huxley batch simulation.

Demonstrates GPU acceleration by simulating multiple neurons with
different stimulus intensities in parallel.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("ERROR: CuPy is not installed!")
    print("Please install: pip install cupy-cuda11x (or cuda12x)")
    sys.exit(1)

from gpu_backed import GPUSimulator
from hh_core.models import HHParameters


def demo_single_neuron():
    """Demonstrate GPU simulation of a single neuron."""
    print("="*70)
    print("GPU Demo: Single Neuron Simulation")
    print("="*70)
    
    # Simulation parameters
    T = 100.0  # ms
    dt = 0.01  # ms
    
    print(f"\nSimulation: {T}ms at dt={dt}ms")
    
    # Create stimulus
    n_steps = int(T / dt) + 1
    stimulus = np.zeros(n_steps)
    stimulus[int(10/dt):int(40/dt)] = 10.0  # 10 µA/cm² from 10-40ms
    
    # Run simulation
    print("Running GPU simulation...")
    simulator = GPUSimulator()
    result = simulator.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V', 'm', 'h', 'n'])
    
    print(f"✓ Complete! Simulated {result.batch_size} neuron(s)")
    
    # Print summary
    print("\n" + result.summary())
    
    # Get spike information
    spike_count = result.get_spike_count()
    print(f"\nDetected {spike_count} spikes")
    
    if spike_count > 0:
        spike_times = result.get_spike_times()
        print(f"Spike times: {spike_times} ms")
        
        if spike_count > 1:
            isis = np.diff(spike_times)
            print(f"Inter-spike intervals: {isis} ms")
            print(f"Mean ISI: {np.mean(isis):.2f} ms")
            firing_rate = 1000.0 / np.mean(isis)  # Convert to Hz
            print(f"Mean firing rate: {firing_rate:.2f} Hz")
    
    # Plot results using the result.plot() method
    print("\nGenerating plots...")
    fig, axes = result.plot(variables=['V', 'm', 'h', 'n'], figsize=(12, 10))
    
    # Add stimulus to voltage plot
    # Only pad if stimulus is shorter than time array
    if len(stimulus) < len(result.time):
        stimulus_padded = np.pad(stimulus, (0, len(result.time) - len(stimulus)), mode='edge')
    else:
        stimulus_padded = stimulus[:len(result.time)]
    axes[0].plot(result.time, stimulus_padded * 5 - 80, 'r--', alpha=0.5, label='Stimulus (scaled)')
    axes[0].legend()
    axes[0].set_title('GPU-Accelerated: Membrane Potential and Gating Variables')
    
    plt.savefig('plots/gpu_single_neuron_demo.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved as 'plots/gpu_single_neuron_demo.png'")
    
    # Show voltage trace details
    V_data = result.V.flatten() if result.V.ndim == 2 and result.V.shape[1] == 1 else result.V
    print("\nVoltage statistics:")
    print(f"  Min: {V_data.min():.2f} mV")
    print(f"  Max: {V_data.max():.2f} mV")
    print(f"  Resting (initial): {V_data[0]:.2f} mV")
    print(f"  Final: {V_data[-1]:.2f} mV")


def demo_batch_simulation():
    """Demonstrate GPU batch simulation with multiple neurons."""
    print("\n" + "="*60)
    print("Batch Simulation Demo - Multiple Neurons")
    print("="*60)
    print()
    
    # Parameters
    T = 100.0  # ms
    dt = 0.01  # ms
    batch_size = 5
    
    print(f"Simulating {batch_size} neurons simultaneously...")
    print()
    
    # Create different stimuli for each neuron
    n_steps = int(T / dt) + 1
    time = np.linspace(0, T, n_steps)
    stimulus = np.zeros((n_steps, batch_size))
    
    # Each neuron gets a different current amplitude
    for i in range(batch_size):
        current = 5.0 + i * 2.0  # 5, 7, 9, 11, 13 µA/cm²
        stimulus[int(10/dt):int(40/dt), i] = current
    
    print("Stimulus amplitudes: 5, 7, 9, 11, 13 µA/cm²")
    print()
    
    # Run simulation
    print("Running batch simulation...")
    simulator = GPUSimulator()
    result = simulator.run(T, dt, batch_size=batch_size, stimulus=stimulus, record_vars=['V'])
    
    print("Complete!")
    print()
    
    # Analyze each neuron
    spike_counts = result.get_spike_count()
    print("Spike counts per neuron:")
    for i, count in enumerate(spike_counts):
        print(f"  Neuron {i}: {count} spikes")
    print()
    
    # Plot all neurons
    print("Plotting results...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i in range(batch_size):
        ax.plot(result.time, result.V[:, i], label=f'Neuron {i}', alpha=0.8)
    
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Voltage (mV)')
    ax.set_title('GPU Batch Simulation: 5 Neurons with Different Stimuli')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig('plots/gpu_batch_simulation_demo.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: gpu_batch_simulation_demo.png")
    print()
    
    print("Batch demo complete!")


if __name__ == '__main__':
    # Check GPU
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    print(f"Using GPU: {props['name'].decode()}")
    print()
    
    # Run demos
    demo_single_neuron()
    demo_batch_simulation()
    
    print("\n" + "="*60)
    print("Demos complete!")
    print("="*60)
    
    # Show plots if running interactively
    try:
        plt.show()
    except:
        pass

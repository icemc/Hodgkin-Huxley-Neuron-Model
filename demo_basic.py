"""
Example: Basic Hodgkin-Huxley simulation

This script demonstrates how to use the CPU-backed HH implementation
to simulate a single neuron with step current injection.
"""

import numpy as np
import matplotlib.pyplot as plt

from hh_optimized import HHModel, Simulator, Stimulus


def basic_demo():
    """Run basic HH simulation demo."""
    
    print("=" * 60)
    print("Hodgkin-Huxley Neuron Simulation - Basic Demo")
    print("=" * 60)
    print()
    
    # Simulation parameters
    T = 100.0  # Total time (ms)
    dt = 0.01  # Time step (ms)
    
    print(f"Simulation duration: {T} ms")
    print(f"Time step: {dt} ms")
    print()
    
    # Create model with default parameters
    model = HHModel()
    print("Model parameters:")
    params = model.get_params()
    print(f"  C_m = {params.C_m} µF/cm²")
    print(f"  g_Na = {params.g_Na} mS/cm²")
    print(f"  g_K = {params.g_K} mS/cm²")
    print(f"  g_L = {params.g_L} mS/cm²")
    print(f"  E_Na = {params.E_Na} mV")
    print(f"  E_K = {params.E_K} mV")
    print(f"  E_L = {params.E_L} mV")
    print()
    
    # Create stimulus: step current from 10 to 40 ms
    print("Creating step current stimulus...")
    stimulus = Stimulus.step(
        amplitude=10.0,  # µA/cm²
        t_start=10.0,    # ms
        t_end=40.0,      # ms
        duration=T,
        dt=dt
    )
    print(f"  Amplitude: 10 µA/cm²")
    print(f"  Duration: 10 to 40 ms")
    print()
    
    # Create simulator with CPU backend and RK4 integrator
    print("Creating simulator (CPU backend, RK4 integrator)...")
    simulator = Simulator(
        model=model,
        backend='cpu',
        integrator='rk4',
        dtype=np.float64
    )
    print()
    
    # Run simulation
    print("Running simulation...")
    result = simulator.run(
        T=T,
        dt=dt,
        stimulus=stimulus,
        batch_size=1,
        spike_threshold=0.0
    )
    print("Simulation complete!")
    print()
    
    # Print summary
    print(result.summary())
    print()
    
    # Get spike information
    spike_count = result.get_spike_count()
    print(f"Detected {spike_count} spikes")
    
    if spike_count > 0:
        spike_times = result.get_spike_times()
        print(f"Spike times: {spike_times} ms")
        
        if spike_count > 1:
            isis = np.diff(spike_times)
            print(f"Inter-spike intervals: {isis} ms")
            print(f"Mean ISI: {np.mean(isis):.2f} ms")
            firing_rate = 1000.0 / np.mean(isis)  # Convert to Hz
            print(f"Mean firing rate: {firing_rate:.2f} Hz")
    print()
    
    # Plot results
    print("Generating plots...")
    fig, axes = result.plot(variables=['V', 'm', 'h', 'n'], figsize=(12, 10))
    
    # Add stimulus to voltage plot (pad stimulus to match time array length)
    stimulus_padded = np.pad(stimulus, (0, 1), mode='edge')
    axes[0].plot(result.time, stimulus_padded * 5 - 80, 'r--', alpha=0.5, label='Stimulus (scaled)')
    axes[0].legend()
    axes[0].set_title('Membrane Potential and Gating Variables')

    plt.savefig('plots/hh_simulation_basic.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: hh_simulation_basic.png")
    print()
    
    # Show voltage trace details
    print("Voltage statistics:")
    print(f"  Min: {result.V.min():.2f} mV")
    print(f"  Max: {result.V.max():.2f} mV")
    print(f"  Resting (initial): {result.V[0]:.2f} mV")
    print(f"  Final: {result.V[-1]:.2f} mV")
    print()
    
    print("Demo complete!")


def batch_simulation_demo():
    """Demonstrate batch simulation with multiple neurons."""
    
    print()
    print("=" * 60)
    print("Batch Simulation Demo - Multiple Neurons")
    print("=" * 60)
    print()
    
    # Parameters
    T = 100.0
    dt = 0.01
    batch_size = 5
    
    print(f"Simulating {batch_size} neurons simultaneously...")
    print()
    
    # Create model and simulator
    model = HHModel()
    simulator = Simulator(model=model, backend='cpu', integrator='rk4')
    
    # Create different stimuli for each neuron
    n_steps = int(np.ceil(T / dt))
    stimulus = np.zeros((n_steps, batch_size))
    
    # Different step currents for each neuron
    for i in range(batch_size):
        amplitude = 5.0 + i * 2.0  # 5, 7, 9, 11, 13 µA/cm²
        stim = Stimulus.step(amplitude, 10.0, 40.0, T, dt)
        stimulus[:, i] = stim
    
    print("Stimulus amplitudes: 5, 7, 9, 11, 13 µA/cm²")
    print()
    
    # Run simulation
    print("Running batch simulation...")
    result = simulator.run(T=T, dt=dt, stimulus=stimulus, batch_size=batch_size)
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
    ax.set_title('Batch Simulation: 5 Neurons with Different Stimuli')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.savefig('plots/hh_simulation_batch.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: hh_simulation_batch.png")
    print()
 
    # Plot 3D phase space n (x), m (y), h (z) for each neuron
    print("Plotting 3D phase space (n vs m vs h) for each neuron...")
    # local import to avoid modifying top-level imports for scripts that don't need 3D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111, projection='3d')

    for i in range(batch_size):
        # plot trajectory in 3D phase space: n (x), m (y), h (z)
        ax2.plot(result.n[:, i], result.m[:, i], result.h[:, i], label=f'Neuron {i}', alpha=0.8)
        # mark start and end points for clarity
        ax2.scatter(result.n[0, i], result.m[0, i], result.h[0, i], marker='o', color='k', s=20)
        ax2.scatter(result.n[-1, i], result.m[-1, i], result.h[-1, i], marker='x', color='k', s=20)

    ax2.set_xlabel('n (activation variable)')
    ax2.set_ylabel('m (activation variable)')
    ax2.set_zlabel('h (inactivation variable)')
    ax2.set_title('3D Phase Space: n (x) vs m (y) vs h (z)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.savefig('plots/hh_simulation_phase_nmh.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: hh_simulation_phase_nmh.png")
    print()

    print("Batch demo complete!")


if __name__ == '__main__':
    # Run basic demo
    basic_demo()
    
    # Run batch demo
    batch_simulation_demo()
    
    # Show plots if running interactively
    try:
        plt.show()
    except:
        pass

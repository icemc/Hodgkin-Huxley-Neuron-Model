"""
Benchmark script: CPU vs GPU performance comparison.

Compares the performance of CPU (NumPy) and GPU (CuPy) implementations
across different batch sizes to demonstrate when GPU acceleration is beneficial.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys

# Check GPU availability
try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU benchmarking enabled")
except ImportError:
    GPU_AVAILABLE = False
    print("✗ CuPy not found - GPU benchmarking disabled")
    print("  Install with: pip install cupy-cuda11x")

from cpu_backed import Simulator, HHModel
if GPU_AVAILABLE:
    from gpu_backed import GPUSimulator
from hh_core.models import HHParameters


def benchmark_cpu(T, dt, batch_size, n_runs=5):
    """Benchmark CPU implementation."""
    model = HHModel()
    simulator = Simulator(model=model, backend='cpu', integrator='rk4', dtype=np.float64)
    
    # Warmup
    _ = simulator.run(T, dt, batch_size=batch_size, record=['V'])
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = simulator.run(T, dt, batch_size=batch_size, record=['V'])
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times)
    }


def benchmark_gpu(T, dt, batch_size, n_runs=5):
    """Benchmark GPU implementation."""
    if not GPU_AVAILABLE:
        return None
    
    simulator = GPUSimulator(dtype='float32')
    result = simulator.benchmark(T, dt, batch_size, n_runs)
    
    return {
        'mean': result['mean_time'],
        'std': result['std_time'],
        'min': result['min_time']
    }


def run_comprehensive_benchmark(T=50.0, dt=0.01, batch_sizes=None, n_runs=5):
    """
    Run comprehensive CPU vs GPU benchmark.
    
    Args:
        T: Simulation time (ms)
        dt: Time step (ms)
        batch_sizes: List of batch sizes to test
        n_runs: Number of runs per benchmark
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000]
    
    print("=" * 80)
    print("Hodgkin-Huxley: CPU vs GPU Performance Benchmark")
    print("=" * 80)
    print(f"\nSimulation parameters:")
    print(f"  Duration: {T} ms")
    print(f"  Time step: {dt} ms")
    print(f"  Steps: {int(T/dt)}")
    print(f"  Runs per test: {n_runs}")
    print(f"\nBatch sizes: {batch_sizes}")
    
    if GPU_AVAILABLE:
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        print(f"\nGPU: {props['name'].decode()}")
    
    print(f"\n{'='*80}\n")
    
    # Store results
    cpu_results = []
    gpu_results = []
    
    # Run benchmarks
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size:>6} neurons", end=" ... ")
        sys.stdout.flush()
        
        # CPU benchmark
        cpu_result = benchmark_cpu(T, dt, batch_size, n_runs)
        cpu_results.append(cpu_result)
        
        # GPU benchmark
        if GPU_AVAILABLE:
            gpu_result = benchmark_gpu(T, dt, batch_size, n_runs)
            gpu_results.append(gpu_result)
            
            speedup = cpu_result['mean'] / gpu_result['mean']
            print(f"CPU: {cpu_result['mean']:.4f}s | GPU: {gpu_result['mean']:.4f}s | Speedup: {speedup:.2f}x")
        else:
            print(f"CPU: {cpu_result['mean']:.4f}s | GPU: N/A")
    
    print(f"\n{'='*80}")
    
    # Create visualization
    plot_results(batch_sizes, cpu_results, gpu_results, T, dt)
    
    # Print summary table
    print_summary_table(batch_sizes, cpu_results, gpu_results)
    
    return batch_sizes, cpu_results, gpu_results


def plot_results(batch_sizes, cpu_results, gpu_results, T, dt):
    """Create performance comparison plots."""
    
    cpu_times = [r['mean'] for r in cpu_results]
    cpu_stds = [r['std'] for r in cpu_results]
    
    if gpu_results and gpu_results[0] is not None:
        gpu_times = [r['mean'] for r in gpu_results]
        gpu_stds = [r['std'] for r in gpu_results]
        speedups = [cpu_results[i]['mean'] / gpu_results[i]['mean'] 
                   for i in range(len(batch_sizes))]
    else:
        gpu_times = None
        speedups = None
    
    # Create figure with subplots
    if gpu_times:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))
        axes = [axes]
    
    # Plot 1: Execution time
    ax = axes[0]
    ax.errorbar(batch_sizes, cpu_times, yerr=cpu_stds, 
               marker='o', linewidth=2, markersize=8, capsize=5,
               label='CPU (NumPy)', color='#2E86AB')
    
    if gpu_times:
        ax.errorbar(batch_sizes, gpu_times, yerr=gpu_stds,
                   marker='s', linewidth=2, markersize=8, capsize=5,
                   label='GPU (CuPy)', color='#A23B72')
    
    ax.set_xlabel('Number of Neurons (Batch Size)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'CPU vs GPU Performance\n({T}ms simulation, dt={dt}ms)', 
                fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.legend(fontsize=11, loc='upper left')
    
    if gpu_times:
        # Plot 2: Speedup
        ax = axes[1]
        ax.plot(batch_sizes, speedups, marker='D', linewidth=2.5, markersize=8,
               color='#F18F01', label='GPU Speedup')
        ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5, label='Break-even')
        
        ax.set_xlabel('Number of Neurons (Batch Size)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (CPU time / GPU time)', fontsize=12, fontweight='bold')
        ax.set_title('GPU Speedup Factor', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.legend(fontsize=11)
        
        # Annotate crossover point
        for i, (bs, sp) in enumerate(zip(batch_sizes, speedups)):
            if sp >= 1.0 and (i == 0 or speedups[i-1] < 1.0):
                ax.annotate(f'Crossover\n~{bs} neurons', 
                           xy=(bs, sp), xytext=(bs*0.5, sp*2),
                           fontsize=10, fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='red', lw=2),
                           bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
                break
        
        # Plot 3: Throughput
        ax = axes[2]
        n_steps = int(T / dt)
        cpu_throughput = [(n_steps * bs) / cpu_results[i]['mean'] 
                         for i, bs in enumerate(batch_sizes)]
        gpu_throughput = [(n_steps * bs) / gpu_results[i]['mean']
                         for i, bs in enumerate(batch_sizes)]
        
        ax.plot(batch_sizes, cpu_throughput, marker='o', linewidth=2, markersize=8,
               label='CPU', color='#2E86AB')
        ax.plot(batch_sizes, gpu_throughput, marker='s', linewidth=2, markersize=8,
               label='GPU', color='#A23B72')
        
        ax.set_xlabel('Number of Neurons (Batch Size)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Throughput (neuron-steps/second)', fontsize=12, fontweight='bold')
        ax.set_title('Computational Throughput', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both', linestyle='--')
        ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    # Save figure
    filename = 'plots/cpu_vs_gpu_benchmark.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved as '{filename}'")
    
    plt.show()


def print_summary_table(batch_sizes, cpu_results, gpu_results):
    """Print summary table of results."""
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    
    if gpu_results and gpu_results[0] is not None:
        print(f"{'Neurons':<10} {'CPU Time (s)':<15} {'GPU Time (s)':<15} {'Speedup':<12} {'Winner':<10}")
        print("-"*100)
        
        for i, bs in enumerate(batch_sizes):
            cpu_time = cpu_results[i]['mean']
            gpu_time = gpu_results[i]['mean']
            speedup = cpu_time / gpu_time
            winner = "GPU" if speedup > 1.0 else "CPU"
            
            print(f"{bs:<10} {cpu_time:<15.4f} {gpu_time:<15.4f} {speedup:<12.2f}x {winner:<10}")
    else:
        print(f"{'Neurons':<10} {'CPU Time (s)':<15}")
        print("-"*50)
        
        for i, bs in enumerate(batch_sizes):
            cpu_time = cpu_results[i]['mean']
            print(f"{bs:<10} {cpu_time:<15.4f}")
    
    print("="*100)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark CPU vs GPU HH simulation')
    parser.add_argument('--T', type=float, default=50.0,
                       help='Simulation time in ms (default: 50.0)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step in ms (default: 0.01)')
    parser.add_argument('--batch-sizes', type=int, nargs='+',
                       default=[1, 10, 50, 100, 500, 1000, 5000, 10000],
                       help='List of batch sizes to test')
    parser.add_argument('--n-runs', type=int, default=5,
                       help='Number of runs per benchmark (default: 5)')
    
    args = parser.parse_args()
    
    run_comprehensive_benchmark(
        T=args.T,
        dt=args.dt,
        batch_sizes=args.batch_sizes,
        n_runs=args.n_runs
    )

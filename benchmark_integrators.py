"""
Benchmark script: RK4 vs RK4-Scipy integrator comparison.

Compares the performance of hand-written RK4 and scipy's RK45 integrator
across different batch sizes.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

from cpu_backed import Simulator, HHModel


def benchmark_integrator(integrator_type, T, dt, batch_size, n_runs=5):
    """
    Benchmark a specific integrator.
    
    Args:
        integrator_type: 'rk4' or 'rk4-scipy'
        T: Simulation time (ms)
        dt: Time step (ms)
        batch_size: Number of neurons
        n_runs: Number of runs for averaging
    
    Returns:
        Dictionary with timing statistics
    """
    model = HHModel()
    simulator = Simulator(model=model, backend='cpu', integrator=integrator_type, dtype=np.float64)
    
    # Create stimulus (constant current)
    n_steps = int(T / dt) + 1
    stim = np.full(n_steps, 10.0, dtype=np.float64)
    
    # Warmup run
    _ = simulator.run(T, dt, batch_size=batch_size, stimulus=stim, record=['V'])
    
    # Benchmark runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        _ = simulator.run(T, dt, batch_size=batch_size, stimulus=stim, record=['V'])
        end = time.perf_counter()
        times.append(end - start)
    
    times = np.array(times)
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times)
    }


def run_comprehensive_benchmark(T=50.0, dt=0.01, batch_sizes=None, n_runs=5):
    """
    Run comprehensive integrator comparison benchmark.
    
    Args:
        T: Simulation time (ms)
        dt: Time step (ms)
        batch_sizes: List of batch sizes to test
        n_runs: Number of runs per benchmark
    """
    if batch_sizes is None:
        batch_sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000]
    
    print("=" * 80)
    print("Hodgkin-Huxley: RK4 vs RK4-Scipy Integrator Benchmark")
    print("=" * 80)
    print(f"\nSimulation parameters:")
    print(f"  Duration: {T} ms")
    print(f"  Time step: {dt} ms")
    print(f"  Steps: {int(T/dt)}")
    print(f"  Runs per test: {n_runs}")
    print(f"\nBatch sizes: {batch_sizes}")
    print(f"\n{'='*80}\n")
    
    # Check if scipy is available
    try:
        from scipy.integrate import RK45
        scipy_available = True
    except ImportError:
        print("WARNING: scipy not installed. Install with: pip install scipy")
        scipy_available = False
        return
    
    # Store results
    rk4_results = []
    scipy_results = []
    
    # Run benchmarks
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size:>6} neurons", end=" ... ")
        sys.stdout.flush()
        
        # Hand-written RK4 benchmark
        rk4_result = benchmark_integrator('rk4', T, dt, batch_size, n_runs)
        rk4_results.append(rk4_result)
        
        # Scipy RK4 benchmark
        scipy_result = benchmark_integrator('rk4-scipy', T, dt, batch_size, n_runs)
        scipy_results.append(scipy_result)
        
        speedup = scipy_result['mean'] / rk4_result['mean']
        speedup_str = f"{speedup:.2f}x"
        if speedup > 1:
            winner = "RK4 faster"
        else:
            winner = "Scipy faster"
            speedup_str = f"{1/speedup:.2f}x"
        
        print(f"RK4: {rk4_result['mean']:.4f}s | Scipy: {scipy_result['mean']:.4f}s | {winner} ({speedup_str})")
    
    print(f"\n{'='*80}")
    
    # Create visualization
    plot_results(batch_sizes, rk4_results, scipy_results, T, dt)
    
    # Print summary table
    print_summary_table(batch_sizes, rk4_results, scipy_results)
    
    return batch_sizes, rk4_results, scipy_results


def plot_results(batch_sizes, rk4_results, scipy_results, T, dt):
    """Create performance comparison plots."""
    
    rk4_times = [r['mean'] for r in rk4_results]
    rk4_stds = [r['std'] for r in rk4_results]
    scipy_times = [r['mean'] for r in scipy_results]
    scipy_stds = [r['std'] for r in scipy_results]
    
    # Calculate speedup (values > 1 mean RK4 is faster, < 1 mean Scipy is faster)
    speedups = [scipy_times[i] / rk4_times[i] for i in range(len(batch_sizes))]
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Absolute times (log scale)
    ax = axes[0]
    ax.errorbar(batch_sizes, rk4_times, yerr=rk4_stds, 
                marker='o', label='RK4 (Hand-written)', linewidth=2, capsize=4)
    ax.errorbar(batch_sizes, scipy_times, yerr=scipy_stds,
                marker='s', label='RK4-Scipy', linewidth=2, capsize=4)
    ax.set_xlabel('Batch Size (neurons)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Absolute Runtime Comparison', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Absolute times (linear scale)
    ax = axes[1]
    ax.errorbar(batch_sizes, rk4_times, yerr=rk4_stds,
                marker='o', label='RK4 (Hand-written)', linewidth=2, capsize=4)
    ax.errorbar(batch_sizes, scipy_times, yerr=scipy_stds,
                marker='s', label='RK4-Scipy', linewidth=2, capsize=4)
    ax.set_xlabel('Batch Size (neurons)', fontsize=12)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Runtime Comparison (Linear Scale)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Speedup ratio
    ax = axes[2]
    colors = ['green' if s > 1 else 'red' for s in speedups]
    ax.bar(range(len(batch_sizes)), speedups, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, label='Equal performance')
    ax.set_xlabel('Batch Size (neurons)', fontsize=12)
    ax.set_ylabel('Speedup (Scipy/RK4)', fontsize=12)
    ax.set_title('Relative Performance\n(>1 = RK4 faster, <1 = Scipy faster)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(bs) for bs in batch_sizes], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add speedup values on bars
    for i, (bs, speedup) in enumerate(zip(batch_sizes, speedups)):
        label = f'{speedup:.2f}x'
        y_pos = speedup + 0.02 if speedup > 1 else speedup - 0.02
        va = 'bottom' if speedup > 1 else 'top'
        ax.text(i, y_pos, label, ha='center', va=va, fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = 'plots'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'integrator_benchmark.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def print_summary_table(batch_sizes, rk4_results, scipy_results):
    """Print formatted summary table."""
    print("\nSummary Table:")
    print("=" * 90)
    print(f"{'Batch Size':<12} {'RK4 (s)':<12} {'Scipy (s)':<12} {'Faster':<15} {'Speedup':<12}")
    print("=" * 90)
    
    for i, batch_size in enumerate(batch_sizes):
        rk4_time = rk4_results[i]['mean']
        scipy_time = scipy_results[i]['mean']
        
        if rk4_time < scipy_time:
            faster = "RK4"
            speedup = scipy_time / rk4_time
        else:
            faster = "Scipy"
            speedup = rk4_time / scipy_time
        
        print(f"{batch_size:<12} {rk4_time:<12.4f} {scipy_time:<12.4f} {faster:<15} {speedup:<12.2f}x")
    
    print("=" * 90)
    
    # Overall summary
    total_rk4 = sum(r['mean'] for r in rk4_results)
    total_scipy = sum(r['mean'] for r in scipy_results)
    
    print(f"\nTotal time across all batch sizes:")
    print(f"  RK4 (Hand-written): {total_rk4:.4f}s")
    print(f"  RK4-Scipy:          {total_scipy:.4f}s")
    
    if total_rk4 < total_scipy:
        print(f"  Overall winner: RK4 (Hand-written) by {total_scipy/total_rk4:.2f}x")
    else:
        print(f"  Overall winner: RK4-Scipy by {total_rk4/total_scipy:.2f}x")


def main():
    """Main benchmark execution."""
    # Default parameters
    T = 50.0  # ms
    dt = 0.01  # ms
    batch_sizes = [1, 10, 50, 100, 500, 1000, 5000, 10000]
    n_runs = 5
    
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help']:
            print("Usage: python benchmark_integrators.py [T] [dt] [n_runs]")
            print("  T: Simulation time in ms (default: 50.0)")
            print("  dt: Time step in ms (default: 0.01)")
            print("  n_runs: Number of runs per benchmark (default: 5)")
            return
        
        if len(sys.argv) > 1:
            T = float(sys.argv[1])
        if len(sys.argv) > 2:
            dt = float(sys.argv[2])
        if len(sys.argv) > 3:
            n_runs = int(sys.argv[3])
    
    # Run benchmark
    run_comprehensive_benchmark(T, dt, batch_sizes, n_runs)


if __name__ == '__main__':
    main()

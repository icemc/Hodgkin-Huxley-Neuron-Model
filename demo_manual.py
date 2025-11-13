"""Demo for the manual HH implementation using the same inputs as demo_basic.py.

Runs a single neuron simulation with a step stimulus (10 -> 40 ms, amplitude 10)
and displays a 4-panel plot (V, m, h, n) with stimulus overlay.
"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from hh_manual import HHManual
from hh_optimized import Stimulus


def run_sim(method, repeats, T, dt):
    # validate method
    if method not in ('rk4', 'euler'):
        raise ValueError("method must be 'rk4' or 'euler' for run_sim")

    sim = HHManual()
    stim = np.asarray(Stimulus.step(10.0, 10.0, 40.0, T, dt), dtype=float)

    # Warm-up (single run to prime caches / any internal state)
    sim.run(T=T, dt=dt, stimulus=stim, method=method)

    t0 = time.perf_counter()
    out = None
    for _ in range(max(1, int(repeats))):
        out = sim.run(T=T, dt=dt, stimulus=stim, method=method)
    t1 = time.perf_counter()

    elapsed = (t1 - t0) / max(1, int(repeats))
    return out, elapsed, stim


def plot_results(out, stim, title, png_path, overlay=None):
    time_arr = out['time']
    V = out['V']
    m = out['m']
    h = out['h']
    n = out['n']

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time_arr, V, label=f'V ({title})', color='C0')
    stim_padded = np.pad(stim, (0, max(0, len(time_arr) - len(stim))), mode='edge')
    axes[0].plot(time_arr, stim_padded * 5 - 80, 'r--', alpha=0.5, label='Stimulus (scaled)')
    if overlay is not None:
        axes[0].plot(time_arr, overlay['V'], label=f'V ({overlay["label"]})', color='C1', alpha=0.8)
    axes[0].set_ylabel('V (mV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_arr, m, label='m')
    axes[1].set_ylabel('m')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_arr, h, label='h')
    axes[2].set_ylabel('h')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time_arr, n, label='n')
    axes[3].set_ylabel('n')
    axes[3].set_xlabel('Time (ms)')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle(f'Manual Hodgkin-Huxley ({title})')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print('Saved plot to', png_path)

    try:
        plt.show()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description='Manual HH benchmark (RK4 vs Euler)')
    parser.add_argument('--method', choices=['rk4', 'euler', 'compare'], default='compare')
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--T', type=float, default=100.0)
    parser.add_argument('--repeats', type=int, default=3, help='Number of repeats for timing')
    args = parser.parse_args()

    os.makedirs('plots', exist_ok=True)

    if args.method in ('rk4', 'euler'):
        print(f'Running method={args.method} (repeats={args.repeats})')
        out, elapsed, stim = run_sim(args.method, args.repeats, args.T, args.dt)
        print(f'Average time per run: {elapsed*1000:.3f} ms')
        png = f'plots/hh_manual_{args.method}_plot.png'
        plot_results(out, stim, args.method.upper(), png)

    else:
        print('Comparing RK4 vs Euler')
        out_rk4, t_rk4, stim = run_sim('rk4', args.repeats, args.T, args.dt)
        out_euler, t_euler, _ = run_sim('euler', args.repeats, args.T, args.dt)

        print(f'RK4 avg time: {t_rk4*1000:.3f} ms')
        print(f'Euler avg time: {t_euler*1000:.3f} ms')

        # Overlay voltage traces
        overlay = {'V': out_euler['V'], 'label': 'Euler'}
        png = 'plots/hh_manual_compare_plot.png'
        plot_results(out_rk4, stim, 'RK4', png, overlay=overlay)


if __name__ == '__main__':
    main()

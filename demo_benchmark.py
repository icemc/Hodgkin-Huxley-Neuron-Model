"""
Combined benchmark: manual vs basic (optimized) Hodgkin-Huxley implementations.

This script runs the manual (HHManual) and the optimized `hh_optimized` Simulator
for the same stimulus, measures average runtime over N repeats, and saves
overlay plots for visual comparison.

Usage examples (use the repo virtualenv to run if you need numpy/matplotlib):

./environment/bin/python3 demo_benchmark.py --implementations all --method rk4 --repeats 3

"""
import os
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from hh_manual import HHManual
from hh_optimized import HHModel, Simulator, Stimulus


def run_manual(method, repeats, T, dt, stim = None):
    sim = HHManual()
    if stim is None:
        stim = Stimulus.step(10.0, 10.0, 40.0, T, dt)

    # warm-up
    sim.run(T=T, dt=dt, stimulus=stim, method=method)

    t0 = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = sim.run(T=T, dt=dt, stimulus=stim, method=method)
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / repeats
    return out, elapsed, stim


def run_basic(method, repeats, T, dt):
    model = HHModel()
    simulator = Simulator(model=model, backend='cpu', integrator=method)
    stim = Stimulus.step(10.0, 10.0, 40.0, T, dt)

    # warm-up
    simulator.run(T=T, dt=dt, stimulus=stim, batch_size=1)

    t0 = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = simulator.run(T=T, dt=dt, stimulus=stim, batch_size=1)
    t1 = time.perf_counter()
    elapsed = (t1 - t0) / repeats
    return result, elapsed, stim


# plot_overlay removed — plotting consolidated in summary section


def main():
    parser = argparse.ArgumentParser(description='Benchmark HH implementations: manual vs basic')
    parser.add_argument('--implementations', choices=['manual', 'basic', 'all'], default='all')
    parser.add_argument('--method', choices=['rk4', 'euler', 'both'], default='both',
                        help="Choose integrator to run. 'both' runs rk4 and euler.")
    parser.add_argument('--dt', type=float, default=0.01)
    parser.add_argument('--T', type=float, default=100.0)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()

    os.makedirs('plots', exist_ok=True)

    # determine which integrators to run
    methods = [args.method] if args.method in ('rk4', 'euler') else ['rk4', 'euler']

    results = {}

    # initialize placeholders to keep summary logic simple
    manual_out = None
    t_manual = None
    basic_res = None
    t_basic = None

    stim = Stimulus.step(10.0, 10.0, 40.0, args.T, args.dt)

    if args.implementations in ('manual', 'all'):
        for m in methods:
            print(f'Running manual ({m}) x{args.repeats}...')
            out, t_manual, _ = run_manual(m, args.repeats, args.T, args.dt, stim)
            print(f'Manual ({m}) average time per run: {t_manual*1000:.3f} ms')
            results[('manual', m)] = {'out': out, 'time': t_manual}
    else:
        # nothing to run for manual
        pass

    if args.implementations in ('basic', 'all'):
        for m in methods:
            print(f'Running basic ({m}) x{args.repeats}...')
            basic_res, t_basic, _ = run_basic(m, args.repeats, args.T, args.dt)
            print(f'Basic ({m}) average time per run: {t_basic*1000:.3f} ms')
            results[('basic', m)] = {'out': basic_res, 'time': t_basic}


    # Now produce summary plots comparing timings and voltage traces
    if results:
        # timing bar chart
        labels = []
        times_ms = []
        for key, val in results.items():
            impl, m = key
            labels.append(f"{impl}-{m}")
            times_ms.append(val['time'] * 1000.0)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(labels, times_ms, color=['C0', 'C1', 'C2', 'C3'][: len(labels)])
        ax.set_ylabel('Average time per run (ms)')
        ax.set_title('Benchmark timings')
        for i, v in enumerate(times_ms):
            ax.text(i, v * 1.01, f"{v:.1f} ms", ha='center', va='bottom')
        plt.tight_layout()
        times_png = 'plots/hh_benchmark_times.png'
        plt.savefig(times_png, dpi=150, bbox_inches='tight')
        print('Saved timing plot to', times_png)
        plt.show()


        # overlay voltage traces
        # choose a common time base (use first available result time)
        any_out = next(iter(results.values()))['out']
        if isinstance(any_out, dict):
            t_common = any_out['time']
        else:
            t_common = any_out.time

        # build aligned voltage traces for all available results
        aligned_V = {}
        for key, val in results.items():
            out = val['out']
            if isinstance(out, dict):
                t = out['time']
                V = out['V']
            else:
                t = out.time
                V = out.V
                if V.ndim == 2:
                    V = V[:, 0]

            if not np.allclose(t, t_common):
                V = np.interp(t_common, t, V)
            aligned_V[key] = V

        # compute mean absolute differences between versions
        diff_means = {}
        keys = list(aligned_V.keys())
        for k in keys:
            others = [aligned_V[j] for j in keys if j != k]
            if not others:
                diff_means[k] = 0.0
            else:
                diffs = [np.mean(np.abs(aligned_V[k] - o)) for o in others]
                diff_means[k] = float(np.mean(diffs))

        # print comparison table
        print('\nAverage mean absolute voltage differences (mV):')
        for k in keys:
            impl, m = k
            print(f'  {impl}-{m}: {diff_means[k]:.4f} mV (avg vs {len(keys)-1} others)')

        # Create a 2x2 grid with one subplot per implementation-method combination
        order = [('manual', 'rk4'), ('manual', 'euler'), ('basic', 'rk4'), ('basic', 'euler')]
        fig2, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
        axes_flat = axes.flatten()

        for i, key in enumerate(order):
            ax = axes_flat[i]
            if key in aligned_V:
                V = aligned_V[key]
                ax.plot(t_common, V, color=f'C{i}')
                ax.set_title(f'{key[0].capitalize()} - {key[1].upper()}')
                # annotate timing only
                timing = results[key]['time'] * 1000.0
                ax.text(0.98, 0.02, f'{timing:.1f} ms', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=9, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))
            else:
                ax.text(0.5, 0.5, 'Not run', ha='center', va='center', transform=ax.transAxes)
            ax.grid(True, alpha=0.3)

        # common axis labels
        fig2.text(0.5, 0.04, 'Time (ms)', ha='center')
        fig2.text(0.04, 0.5, 'Voltage (mV)', va='center', rotation='vertical')

        overlay_png = 'plots/hh_benchmark_voltages.png'
        plt.tight_layout(rect=[0.03, 0.05, 1, 0.97])
        plt.savefig(overlay_png, dpi=150, bbox_inches='tight')
        print('Saved voltage comparison grid to', overlay_png)
        plt.show()

if __name__ == '__main__':
    main()

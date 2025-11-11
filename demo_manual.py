"""Demo for the manual HH implementation using the same inputs as demo_basic.py.

Runs a single neuron simulation with a step stimulus (10 -> 40 ms, amplitude 10)
and displays a 4-panel plot (V, m, h, n) with stimulus overlay.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

from hh_manual_simple import HHManual


def step(amplitude, t_start, t_end, duration, dt):
    # helper local wrapper to produce array matching hh_manual_simple.step-like behavior
    n_steps = int(np.ceil(duration / dt))
    arr = np.zeros(n_steps)
    start = int(np.round(t_start / dt))
    end = int(np.round(t_end / dt))
    start = max(0, start)
    end = min(n_steps - 1, end)
    arr[start:end + 1] = amplitude
    return arr


def main():
    T = 100.0
    dt = 0.01

    sim = HHManual()

    os.makedirs('plots', exist_ok=True)

    stim = step(10.0, 10.0, 40.0, T, dt)

    print('Running manual HH (RK4) ...')
    out = sim.run(T=T, dt=dt, stimulus=stim, method='rk4')
    print('Done')

    time = out['time']
    V = out['V']
    m = out['m']
    h = out['h']
    n = out['n']
    I = out['I']

    # Plot 4 panels
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(time, V, label='V (mV)')
    stim_padded = np.pad(stim, (0, max(0, len(time) - len(stim))), mode='edge')
    axes[0].plot(time, stim_padded * 5 - 80, 'r--', alpha=0.5, label='Stimulus (scaled)')
    axes[0].set_ylabel('V (mV)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time, m, label='m')
    axes[1].set_ylabel('m')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time, h, label='h')
    axes[2].set_ylabel('h')
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(time, n, label='n')
    axes[3].set_ylabel('n')
    axes[3].set_xlabel('Time (ms)')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle('Manual Hodgkin-Huxley (RK4)')
    plt.tight_layout()
    out_png = os.path.join('plots', 'hh_manual_simple_plot.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    print('Saved plot to', out_png)

    try:
        plt.show()
    except Exception:
        pass


if __name__ == '__main__':
    main()

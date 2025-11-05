# Getting Started with HH Simulation

This guide will help you get up and running with the CPU-backed Hodgkin-Huxley implementation.

## Installation

1. **Install NumPy** (required):
   ```bash
   pip install numpy
   ```

2. **Install optional dependencies** for better performance and visualization:
   ```bash
   pip install numba matplotlib
   ```

   Or install all at once:
   ```bash
   pip install -r requirements.txt
   ```

## Your First Simulation

Create a file called `my_first_hh.py`:

```python
import numpy as np
from hh_optimized import HHModel, Simulator, Stimulus

# Create the neuron model
model = HHModel()

# Create a simulator
simulator = Simulator(model=model, backend='cpu', integrator='rk4')

# Create a step current stimulus
stimulus = Stimulus.step(
    amplitude=10.0,    # Current amplitude in µA/cm²
    t_start=10.0,      # Start time in ms
    t_end=40.0,        # End time in ms
    duration=100.0,    # Total duration in ms
    dt=0.01           # Time step in ms
)

# Run the simulation
result = simulator.run(
    T=100.0,           # Simulation time in ms
    dt=0.01,          # Time step in ms
    stimulus=stimulus
)

# Print results
print(result.summary())
print(f"\nSpikes detected: {result.get_spike_count()}")

# Plot the results (requires matplotlib)
try:
    result.plot()
    import matplotlib.pyplot as plt
    plt.show()
except ImportError:
    print("Install matplotlib to see plots: pip install matplotlib")
```

Run it:
```bash
python my_first_hh.py
```

## What's Happening?

1. **Model Creation**: `HHModel()` creates a neuron with default Hodgkin-Huxley parameters
2. **Simulator Setup**: Chooses the CPU backend with RK4 integrator (4th-order Runge-Kutta)
3. **Stimulus**: A step current that turns on at 10ms and off at 40ms
4. **Simulation**: Integrates the differential equations forward in time
5. **Results**: Contains voltage traces, gating variables, and detected spikes

## Next Steps

### Modify Parameters

```python
model = HHModel()
model.set_params(
    g_Na=100.0,   # Reduce sodium conductance
    g_K=40.0      # Increase potassium conductance
)
```

### Try Different Stimuli

```python
# Constant current
stim = Stimulus.constant(amplitude=8.0, duration=100.0, dt=0.01)

# Pulse train (repeated pulses)
stim = Stimulus.pulse_train(
    amplitude=15.0,
    pulse_duration=2.0,
    pulse_period=10.0,
    n_pulses=5,
    t_start=10.0,
    duration=100.0,
    dt=0.01
)

# Noisy current
stim = Stimulus.noisy(mean=5.0, std=2.0, duration=100.0, dt=0.01, seed=42)
```

### Simulate Multiple Neurons

```python
# Simulate 100 neurons at once
result = simulator.run(
    T=100.0,
    dt=0.01,
    stimulus=stimulus,
    batch_size=100  # <-- Key parameter
)

# Results now have shape (n_steps, 100)
print(result.V.shape)

# Get spike count for each neuron
spike_counts = result.get_spike_count()
print(f"Average spikes: {np.mean(spike_counts)}")
```

### Use Faster Numba Backend (for single neurons)

```python
# Requires: pip install numba
simulator = Simulator(model=model, backend='numba', integrator='rk4')
result = simulator.run(T=100.0, dt=0.01, stimulus=stimulus)
```

This can be 5-10x faster for single neuron simulations!

### Access Individual Variables

```python
result = simulator.run(...)

# Time array
t = result.time

# Membrane potential
V = result.V

# Gating variables
m = result.m  # Sodium activation
h = result.h  # Sodium inactivation  
n = result.n  # Potassium activation

# Spike information
spike_count = result.get_spike_count()
spike_times = result.get_spike_times()
```

### Record Only What You Need

```python
# Only record voltage (saves memory for large batches)
result = simulator.run(
    T=100.0,
    dt=0.01,
    stimulus=stimulus,
    record=['V']  # <-- Only voltage
)
```

## Common Patterns

### Parameter Sweep

```python
# Test different current amplitudes
amplitudes = np.linspace(0, 20, 21)  # 0 to 20 µA/cm²

spike_counts = []
for amp in amplitudes:
    stim = Stimulus.step(amp, 10.0, 40.0, 100.0, 0.01)
    result = simulator.run(T=100.0, dt=0.01, stimulus=stim)
    spike_counts.append(result.get_spike_count())

# Plot F-I curve
import matplotlib.pyplot as plt
plt.plot(amplitudes, spike_counts)
plt.xlabel('Current (µA/cm²)')
plt.ylabel('Spike Count')
plt.title('F-I Curve')
plt.show()
```

### Long Simulation

```python
# Simulate for 1 second
result = simulator.run(
    T=1000.0,      # 1 second
    dt=0.01,
    stimulus=Stimulus.constant(5.0, 1000.0, 0.01)
)

print(f"Simulated {len(result.time)} time steps")
```

### Different Initial Conditions

```python
from hh_core import HHState
import numpy as np

# Custom initial state (depolarized)
state0 = HHState(np.array([-50.0, 0.1, 0.6, 0.3]))  # [V, m, h, n]

result = simulator.run(
    T=100.0,
    dt=0.01,
    state0=state0,
    stimulus=None  # No external current
)
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Make sure you're in the project directory
cd /path/to/project

# Or add to Python path
export PYTHONPATH=/path/to/project:$PYTHONPATH  # Linux/Mac
set PYTHONPATH=C:\path\to\project;%PYTHONPATH%  # Windows
```

### Numerical Instability

If you see unrealistic results (NaN, very large numbers):
- **Reduce dt**: Try `dt=0.005` instead of `dt=0.01`
- **Use RK4**: Change `integrator='rk4'` (more stable than Euler)
- **Check stimulus**: Make sure current values are reasonable (0-50 µA/cm²)

### No Spikes Detected

If the neuron doesn't spike:
- **Increase current**: Try higher amplitude (>5 µA/cm²)
- **Check duration**: Make sure stimulus is long enough
- **Lower threshold**: Try `spike_threshold=-20.0` instead of `0.0`

### Memory Issues with Large Batches

If simulating thousands of neurons:
- **Record less**: Use `record=['V']` to save memory
- **Use float32**: Set `dtype=np.float32` in Simulator
- **Process in chunks**: Simulate batches separately

## Examples

See `demo_basic.py` for complete examples:
```bash
python demo_basic.py
```

This will create:
- `hh_simulation_basic.png` - Single neuron trace
- `hh_simulation_batch.png` - Multiple neurons

## Performance Tips

| Scenario | Recommended Backend | Expected Speed |
|----------|-------------------|----------------|
| Single neuron | `numba` | ~0.1-1ms per simulation ms |
| Small batch (< 10) | `numba` or `cpu` | ~0.5-5ms per simulation ms |
| Large batch (> 100) | `cpu` | ~1-10ms per simulation ms per neuron |

- RK4 is ~4x slower than Euler but much more accurate
- float32 uses half the memory of float64
- Recording all variables uses 4x memory vs just voltage

## Need Help?

1. Check the full documentation in `README.md`
2. Look at the design document for implementation details
3. Run the test suite: `python test_basic.py`

Happy simulating! 🧠⚡

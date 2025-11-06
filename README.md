# Optimized Hodgkin-Huxley Implementation (Python) - CPU Backend

This is the CPU-optimized implementation of the Hodgkin-Huxley neuron model in Python, as described in the design document.

## Overview

This implementation provides:
- **High-performance vectorized NumPy** implementation for batch simulations
- **Multiple integration methods**: Forward Euler, RK4, and RK4 with Rush-Larsen
- **Flexible stimulus generation**: step, pulse train, ramp, noisy currents
- **Spike detection** with interpolation for precise timing
- **Clean, modular API** for easy use

## Project Structure

```
project/
├── .github/                   # GitHub workflows and CI configuration
├── hh_core/                   # Core HH equations and integrators
│   ├── __init__.py
│   ├── models.py              # HH equations, parameters, gating kinetics
│   ├── integrators.py         # RK4, Euler, Rush-Larsen integrators
│   └── utils.py               # Stimulus generation, spike detection
├── cpu_backed/                # CPU-optimized implementations
│   ├── __init__.py
│   └── vectorized.py          # NumPy vectorized simulator
├── test/                      # Pytest test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures and configuration
│   ├── test_basic.py          # Basic functionality tests
│   └── test_validation.py    # Comprehensive validation tests
├── docs/                      # Additional documentation
├── plots/                     # Generated plots from demos and tests
├── hh_optimized.py            # High-level API (HHModel, Simulator, Stimulus)
├── demo_basic.py              # Example usage script
├── validate.py                # Validation script wrapper
├── pytest.ini                 # Pytest configuration
├── requirements.txt           # Python dependencies
├── TESTING.md                 # Testing documentation
└── README.md                  # This file
```

## Installation

### Required Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Single Neuron Simulation

```python
import numpy as np
from hh_optimized import HHModel, Simulator, Stimulus

# Create model with default parameters
model = HHModel()

# Create simulator (CPU backend, RK4 integrator)
simulator = Simulator(model=model, backend='cpu', integrator='rk4')

# Generate step current stimulus
stimulus = Stimulus.step(
    amplitude=10.0,   # µA/cm²
    t_start=10.0,     # ms
    t_end=40.0,       # ms
    duration=100.0,   # ms
    dt=0.01          # ms
)

# Run simulation
result = simulator.run(
    T=100.0,          # Total time (ms)
    dt=0.01,          # Time step (ms)
    stimulus=stimulus,
    spike_threshold=0.0
)

# Access results
print(result.summary())
print(f"Spikes detected: {result.get_spike_count()}")

# Plot results
fig, axes = result.plot(variables=['V', 'm', 'h', 'n'])
```

### Batch Simulation (Multiple Neurons)

```python
# Simulate 100 neurons with same stimulus
batch_size = 100
result = simulator.run(
    T=100.0,
    dt=0.01,
    stimulus=stimulus,
    batch_size=batch_size
)

# Voltage shape: (n_steps, batch_size)
print(result.V.shape)
```

## Features

### Model Parameters

Default Hodgkin-Huxley parameters (from 1952 paper):
- `C_m = 1.0` µF/cm² (membrane capacitance)
- `g_Na = 120.0` mS/cm² (sodium conductance)
- `g_K = 36.0` mS/cm² (potassium conductance)
- `g_L = 0.3` mS/cm² (leak conductance)
- `E_Na = 50.0` mV (sodium reversal potential)
- `E_K = -77.0` mV (potassium reversal potential)
- `E_L = -54.387` mV (leak reversal potential)

Modify parameters:
```python
model = HHModel()
model.set_params(g_Na=100.0, E_K=-80.0)
```

### Integrators

1. **Forward Euler** (`integrator='euler'`)
   - Simple, first-order
   - Fast but less accurate
   - May require smaller dt

2. **RK4** (`integrator='rk4'`) - **Recommended**
   - Fourth-order Runge-Kutta
   - Excellent accuracy/performance tradeoff
   - Standard choice for HH simulations

3. **RK4 with Rush-Larsen** (`integrator='rk4rl'`)
   - RK4 for voltage, Rush-Larsen for gating variables
   - More stable, can use larger dt
   - Good for stiff systems

### Stimulus Types

```python
from hh_optimized import Stimulus

# Constant current
stim = Stimulus.constant(amplitude=10.0, duration=100.0, dt=0.01)

# Step current
stim = Stimulus.step(amplitude=10.0, t_start=10.0, t_end=40.0, 
                     duration=100.0, dt=0.01)

# Pulse train
stim = Stimulus.pulse_train(amplitude=15.0, pulse_duration=2.0,
                            pulse_period=10.0, n_pulses=5,
                            t_start=10.0, duration=100.0, dt=0.01)

# Noisy current
stim = Stimulus.noisy(mean=5.0, std=2.0, duration=100.0, dt=0.01, seed=42)

# Ramp current
stim = Stimulus.ramp(start_amplitude=0.0, end_amplitude=20.0,
                     duration=100.0, dt=0.01)
```

### Spike Detection

Automatic spike detection using threshold crossing:

```python
result = simulator.run(..., spike_threshold=0.0)

# Get spike count
count = result.get_spike_count()

# Get precise spike times (with interpolation)
spike_times = result.get_spike_times()
```

## Performance Tips

1. **Choose proper dt:**
   - Standard: `dt=0.01` ms
   - Minimum recommended: `dt=0.005` ms
   - Maximum safe: `dt=0.05` ms (with RK4)

2. **Record only needed variables:**
   ```python
   result = simulator.run(..., record=['V'])  # Only voltage
   ```

3. **Use float32 for memory savings:**
   ```python
   simulator = Simulator(..., dtype=np.float32)
   ```

## Examples

Run the included demo:

```bash
python demo_basic.py
```

This will generate two plots:
- `hh_simulation_basic.png` - Single neuron simulation
- `hh_simulation_batch.png` - Batch simulation with 5 neurons

## API Reference

### HHModel

```python
model = HHModel(params=None)  # Create with optional custom parameters
model.set_params(g_Na=100.0)  # Update parameters
params = model.get_params()    # Get HHParameters object
state = model.resting_state(batch_size=1)  # Get resting state
```

### Simulator

```python
simulator = Simulator(
    model=None,           # HHModel (default if None)
    backend='cpu',        # Only 'cpu' backend supported
    integrator='rk4',     # 'euler', 'rk4', or 'rk4rl'
    dtype=np.float64      # np.float32 or np.float64
)

result = simulator.run(
    T,                    # Total time (ms)
    dt=0.01,             # Time step (ms)
    state0=None,         # Initial state (resting if None)
    stimulus=None,       # Current array
    batch_size=1,        # Number of neurons
    record=['V','m','h','n'],  # Variables to record
    spike_threshold=0.0  # Spike detection threshold (mV)
)
```

### SimulationResult

```python
result.time            # Time array
result.V               # Voltage trace
result.m, result.h, result.n  # Gating variables
result.spikes          # Spike detection results

result.get_spike_count()    # Number of spikes
result.get_spike_times()    # Spike times
result.summary()            # Text summary
result.plot()               # Create plots
```

## Testing & Validation

A comprehensive test suite validates the implementation against known HH behavior using pytest.

### Run All Tests
```bash
# Run all tests
pytest test/

# Run with verbose output
pytest test/ -v

# Run specific test categories
pytest test/ -k "physiological"
pytest test/ -k "numerical"

# Run with coverage report
pytest test/ --cov=. --cov-report=html
```

### Test Categories

#### 1. Physiological Behavior Tests
- **Resting Potential**: Verifies neuron settles to ~-65 mV at rest
- **Spike Threshold (Rheobase)**: Confirms threshold current is 4-7 µA/cm²
- **Action Potential Amplitude**: Validates spike peaks reach 30-60 mV
- **F-I Curve**: Tests monotonic increase in firing rate with current

#### 2. Gating Variables Tests
- **Bounds Check**: Ensures m, h, n stay in [0, 1]
- **Resting State Values**: Validates steady-state gating at rest

#### 3. Numerical Accuracy Tests
- **dt Convergence**: Confirms results converge as dt decreases
- **Integrator Consistency**: Compares RK4 vs RK4-Rush-Larsen
- **NaN/Inf Check**: Ensures no numerical errors

#### 4. Batch Simulation Tests
- **Single-Batch Equivalence**: batch_size=1 matches single neuron
- **Independence**: Verifies neurons in batch are independent

#### 5. Reference Comparison Tests
- **SciPy Validation**: Compares against scipy.integrate (requires scipy)

### Example Output:
```bash
$ pytest test/ -v

test/test_validation.py::TestPhysiologicalBehavior::test_resting_potential PASSED
test/test_validation.py::TestPhysiologicalBehavior::test_spike_threshold PASSED
test/test_validation.py::TestPhysiologicalBehavior::test_action_potential_amplitude PASSED
test/test_validation.py::TestPhysiologicalBehavior::test_firing_rate_increases_with_current PASSED
test/test_validation.py::TestGatingVariables::test_gating_variables_stay_in_bounds PASSED
test/test_validation.py::TestNumericalAccuracy::test_dt_convergence PASSED
test/test_validation.py::TestNumericalAccuracy::test_integrator_consistency PASSED
test/test_validation.py::TestBatchSimulation::test_batch_single_equivalence PASSED

===================== 8 passed in 12.34s =====================
```

## Future Extensions (GPU Backend)

The design supports GPU acceleration (Phase B):
- CuPy for CUDA-accelerated arrays
- JAX for XLA compilation and autodiff
- Custom CUDA kernels for maximum performance

These will be implemented in the `gpu_backend/` directory following the same API.

## License

[MIT](./LICENSE)

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). "A quantitative description of membrane current and its application to conduction and excitation in nerve." *The Journal of Physiology*, 117(4), 500-544.
- Rush, S., & Larsen, H. (1978). "A practical algorithm for solving dynamic membrane equations." *IEEE Transactions on Biomedical Engineering*, (4), 389-392.

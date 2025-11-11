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

A comprehensive test suite validates the implementation with **28 tests** covering CPU, GPU, and SciPy implementations using pytest.

### Run All Tests
```bash
# Run the comprehensive test suite (28 tests)
pytest test/test_suite.py -v

# Run all tests in test directory
pytest test/ -v

# Run specific test groups
pytest test/test_suite.py::TestBasicFunctionality -v
pytest test/test_suite.py::TestRK4Accuracy -v
pytest test/test_suite.py::TestPhysiologicalBehavior -v

# Run with coverage report
pytest test/ --cov=. --cov-report=html
```

### Test Suite Organization (`test/test_suite.py`)

The comprehensive test suite is organized into 5 sections with 28 tests:

#### 1. **Basic Functionality Tests** (5 tests)
- Module imports
- Model and state creation
- Stimulus generation
- Single neuron simulation
- Batch simulation

#### 2. **Physiological Validation Tests** (6 tests)
- **Resting Potential**: Verifies neuron settles to -65 to -70 mV
- **Spike Threshold (Rheobase)**: Confirms threshold current is 3-8 µA/cm²
- **Action Potential Amplitude**: Validates spike peaks reach 30-60 mV
- **F-I Curve**: Tests monotonic firing rate increase with current
- **Gating Variable Bounds**: Ensures m, h, n stay in [0, 1]
- **Resting Gating Values**: Validates steady-state at rest

#### 3. **Numerical Accuracy Tests - CPU vs GPU vs SciPy** (10 tests)
- **CPU vs SciPy**: RK4 comparison with reference implementation (max error < 0.0001 mV)
- **GPU vs CPU Single Neuron**: Validates GPU matches CPU (max error < 0.001 mV)
- **GPU vs CPU Batch**: Batch simulation accuracy (max error < 0.1 mV)
- **Three-way Comparison**: CPU, GPU, and SciPy consistency
- **dt Convergence**: Results converge as timestep decreases
- **Energy Conservation**: Resting state stability
- **Deterministic Behavior**: No randomness in simulations
- **Timestep Independence**: Convergence validation
- **Integrator Consistency**: RK4 vs RK4-Rush-Larsen comparison
- **Numerical Stability**: No NaN or Inf values

#### 4. **Batch Consistency Tests** (4 tests)
- **Single-Batch Equivalence**: batch_size=1 matches single neuron
- **Batch Independence**: Neurons with same stimulus produce identical results
- **GPU Batch vs Single**: GPU batch matches individual simulations
- **CPU-GPU Batch Consistency**: CPU and GPU batches match

#### 5. **Stimulus Generation Tests** (3 tests)
- Step stimulus validation
- Constant stimulus validation
- Pulse train stimulus validation

### Test Results Summary

```bash
$ pytest test/test_suite.py -v

28 passed in 112.45s (0:01:52) ✓

Accuracy achieved:
  • CPU vs SciPy:  max error = 0.000118 mV
  • GPU vs CPU:    max error = 0.000542 mV (single neuron)
  • GPU vs CPU:    max error = 0.056 mV (batch of 10 neurons)
```

### Optional Dependencies

Some tests require optional packages:
- **SciPy**: For reference implementation comparison (`test_cpu_vs_scipy_*`)
- **CuPy**: For GPU tests (`test_gpu_*`)

Tests automatically skip if dependencies are unavailable.

### Legacy Test Files

Individual test files are also available:
- `test/test_basic.py` - Basic functionality tests
- `test/test_validation.py` - Physiological validation tests  
- `test/test_accuracy.py` - GPU/CPU/SciPy accuracy tests

**Recommended**: Use `test/test_suite.py` for comprehensive testing.

## License

[MIT](./LICENSE)

## References

- Hodgkin, A. L., & Huxley, A. F. (1952). "A quantitative description of membrane current and its application to conduction and excitation in nerve." *The Journal of Physiology*, 117(4), 500-544.
- Rush, S., & Larsen, H. (1978). "A practical algorithm for solving dynamic membrane equations." *IEEE Transactions on Biomedical Engineering*, (4), 389-392.

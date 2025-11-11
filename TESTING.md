# Testing Guide

This document explains how to run and write tests for the Hodgkin-Huxley implementation.

## Quick Start

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest test/

# Run with verbose output
pytest test/ -v

# Run with coverage
pytest test/ --cov=. --cov-report=html
```

## Test Organization

The project uses a comprehensive test suite that consolidates all tests in a single file:

```
test/
├── __init__.py              # Test package initialization
├── conftest.py              # Shared fixtures and configuration
└── test_suite.py            # Comprehensive test suite (32 tests)
```

The test suite is organized into 7 test classes:
1. **TestBasicFunctionality** - Core functionality and integrators
2. **TestPhysiologicalBehavior** - Biological realism validation
3. **TestGatingVariables** - Gating variable constraints
4. **TestRK4Accuracy** - CPU vs GPU vs SciPy accuracy
5. **TestNumericalProperties** - Numerical stability and convergence
6. **TestBatchConsistency** - Batch simulation correctness
7. **TestStimulusGeneration** - Stimulus generation utilities

## Running Tests

### Basic Usage

```bash
# Run all tests
pytest test/

# Run specific test file
pytest test/test_suite.py

# Run specific test class
pytest test/test_suite.py::TestPhysiologicalBehavior

# Run specific test method
pytest test/test_suite.py::TestPhysiologicalBehavior::test_resting_potential
```

### Filtering Tests

```bash
# Run tests matching pattern
pytest test/ -k "resting"
pytest test/ -k "physiological"
pytest test/ -k "numerical"

# Run tests with specific marker
pytest test/ -m physiological
pytest test/ -m numerical
pytest test/ -m batch
```

### Output Options

```bash
# Verbose output
pytest test/ -v

# Very verbose (show test names as they run)
pytest test/ -vv

# Show print statements
pytest test/ -s

# Show local variables on failure
pytest test/ -l

# Stop at first failure
pytest test/ -x

# Run last failed tests
pytest test/ --lf
```

### Coverage Reports

```bash
# Generate coverage report
pytest test/ --cov=.

# Generate HTML coverage report
pytest test/ --cov=. --cov-report=html

# View coverage report
# Open htmlcov/index.html in browser

# Generate terminal coverage report with missing lines
pytest test/ --cov=. --cov-report=term-missing
```

## Test Categories

### 1. Basic Functionality (`TestBasicFunctionality`)

Tests core functionality and all integrator types:

- `test_imports`: Module import verification
- `test_model_creation`: Model and state creation
- `test_stimulus_generation`: Stimulus generation utilities
- `test_basic_simulation`: Single neuron simulation
- `test_batch_simulation`: Batch simulation
- `test_different_integrators`: All three integrators (euler, rk4, rk4rl)

**Run with:**
```bash
pytest test/test_suite.py::TestBasicFunctionality -v
```

### 2. Physiological Behavior (`TestPhysiologicalBehavior`)

Tests that verify the model produces physiologically realistic behavior:

- `test_resting_potential`: Resting membrane potential ~-65 mV
- `test_spike_threshold`: Rheobase current 4-7 µA/cm²
- `test_action_potential_amplitude`: Spike amplitude and overshoot
- `test_firing_rate_increases_with_current`: F-I curve monotonicity

**Run with:**
```bash
pytest test/test_suite.py::TestPhysiologicalBehavior -v
```

### 3. Gating Variables (`TestGatingVariables`)

Tests for gating variable behavior:

- `test_gating_variables_stay_in_bounds`: m, h, n ∈ [0, 1]
- `test_resting_gating_values`: Steady-state values at rest

**Run with:**
```bash
pytest test/test_suite.py::TestGatingVariables -v
```

### 4. RK4 Accuracy (`TestRK4Accuracy`)

Comprehensive accuracy tests comparing CPU, GPU, and SciPy implementations:

- `test_cpu_vs_scipy_single_neuron`: CPU RK4 vs SciPy reference
- `test_gpu_vs_cpu_single_neuron`: GPU vs CPU single neuron
- `test_gpu_vs_cpu_batch`: GPU vs CPU batch simulation
- `test_all_three_implementations`: Three-way comparison
- `test_scipy_reference_comparison`: Comprehensive SciPy validation

**Run with:**
```bash
pytest test/test_suite.py::TestRK4Accuracy -v
```

### 5. Numerical Properties (`TestNumericalProperties`)

Tests for numerical stability and accuracy:

- `test_dt_convergence`: Results converge as dt decreases
- `test_energy_conservation`: Resting state stability
- `test_deterministic_behavior`: No randomness
- `test_timestep_independence`: Timestep convergence
- `test_integrator_consistency`: Different integrators agree
- `test_no_nans_or_infs`: No numerical errors

**Run with:**
```bash
pytest test/test_suite.py::TestNumericalProperties -v
```

### 6. Batch Consistency (`TestBatchConsistency`)

Tests for batch simulation correctness:

- `test_batch_single_equivalence`: batch_size=1 equals single
- `test_batch_independence`: Neurons are independent
- `test_gpu_batch_matches_single`: GPU batch consistency
- `test_cpu_gpu_batch_consistency`: CPU-GPU batch matching

**Run with:**
```bash
pytest test/test_suite.py::TestBatchConsistency -v
```

### 7. Stimulus Generation (`TestStimulusGeneration`)

Tests for stimulus generation utilities:

- `test_step_stimulus`: Step stimulus generation
- `test_constant_stimulus`: Constant stimulus generation
- `test_pulse_train_stimulus`: Pulse train generation

**Run with:**
```bash
pytest test/test_suite.py::TestStimulusGeneration -v
```

## Configuration

Test configuration is in `pytest.ini`:

```ini
[pytest]
# Test discovery
testpaths = test
python_files = test_*.py
python_classes = Test*
python_functions = test_*

# Output options
addopts = -v --tb=short --strict-markers --color=yes
```

## Fixtures

Common test fixtures are defined in `test/conftest.py`:

- `default_model`: Default HH model
- `cpu_simulator`: CPU simulator with RK4
- `step_stimulus`: Standard step stimulus
- `constant_stimulus`: Constant current stimulus
- `all_integrators`: Parameterized fixture for all integrators

**Using fixtures:**
```python
def test_example(cpu_simulator, step_stimulus):
    result = cpu_simulator.run(T=100.0, dt=0.01, stimulus=step_stimulus)
    assert result.V is not None
```

## Writing New Tests

### Basic Test Structure

```python
import pytest
import numpy as np
from cpu_backed import HHModel, Simulator, Stimulus

class TestMyFeature:
    """Tests for my new feature."""
    
    def test_something(self):
        """Test that something works."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        result = simulator.run(T=100.0, dt=0.01, stimulus=None)
        
        assert result.V is not None
        assert len(result.V) == 10001
```

### Using Fixtures

```python
def test_with_fixtures(cpu_simulator, step_stimulus):
    """Test using pre-configured fixtures."""
    result = cpu_simulator.run(T=100.0, dt=0.01, stimulus=step_stimulus)
    assert result.get_spike_count() > 0
```

### Parametrized Tests

```python
@pytest.mark.parametrize("current,expected_spikes", [
    (5.0, 0),
    (10.0, 2),
    (15.0, 5),
])
def test_spike_counts(cpu_simulator, current, expected_spikes):
    """Test spike counts for different currents."""
    stim = Stimulus.constant(current, 100.0, 0.01)
    result = cpu_simulator.run(T=100.0, dt=0.01, stimulus=stim)
    assert result.get_spike_count() >= expected_spikes
```

### Marking Tests

```python
@pytest.mark.slow
def test_long_simulation(cpu_simulator):
    """Test that takes a long time."""
    result = cpu_simulator.run(T=10000.0, dt=0.01, stimulus=None)
    assert result.V is not None

@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
def test_needs_scipy():
    """Test that requires scipy."""
    from scipy.integrate import solve_ivp
    # ... test code ...
```

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests with coverage
pytest test/ --cov=. --cov-report=xml --cov-report=term

# Check coverage threshold (e.g., 80%)
pytest test/ --cov=. --cov-fail-under=80
```

## Troubleshooting

### Tests Run Slowly

```bash
# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest test/ -n auto
```

### Import Errors

Make sure you're in the project root directory:
```bash
cd /path/to/project
pytest test/
```

Or set PYTHONPATH:
```bash
export PYTHONPATH=/path/to/project:$PYTHONPATH  # Linux/Mac
set PYTHONPATH=C:\path\to\project;%PYTHONPATH%  # Windows
```

### Scipy Tests Skipped

Install scipy:
```bash
pip install scipy
```

### Coverage Not Working

Install pytest-cov:
```bash
pip install pytest-cov
```

## Best Practices

1. **Test Independence**: Each test should be independent
2. **Descriptive Names**: Use clear, descriptive test names
3. **One Concept**: Each test should verify one concept
4. **Use Fixtures**: Reuse common setup via fixtures
5. **Fast Tests**: Keep tests fast; mark slow tests with `@pytest.mark.slow`
6. **Clear Assertions**: Use descriptive assertion messages
7. **Test Edge Cases**: Test boundary conditions and edge cases

## Example: Full Test Session

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run all tests with coverage
pytest test/ -v --cov=. --cov-report=html

# 3. View coverage report
# Open htmlcov/index.html in browser

# 4. Run specific category
pytest test/ -k "physiological" -v

# 5. Run failed tests only
pytest test/ --lf -v

# 6. Generate coverage badge (optional)
coverage-badge -o coverage.svg
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing Best Practices](https://docs.pytest.org/en/latest/goodpractices.html)

"""
Basic functionality tests for the CPU-backed HH implementation.

These tests verify that core functionality works correctly.
Run with: pytest test/test_basic.py
"""

import numpy as np
import pytest


def test_imports():
    """Test that all modules can be imported."""
    from hh_core import HHParameters, HHState, RK4, ForwardEuler
    from hh_core.utils import Stimulus
    from cpu_backed import VectorizedSimulator
    from hh_optimized import HHModel, Simulator
    
    # If we get here, all imports succeeded
    assert True


def test_model_creation():
    """Test model and state creation."""
    from hh_optimized import HHModel
    
    model = HHModel()
    params = model.get_params()
    assert params.C_m == 1.0
    assert params.g_Na == 120.0
    
    state = model.resting_state()
    assert state.data.shape == (4,)
    
    batch_state = model.resting_state(batch_size=10)
    assert batch_state.data.shape == (10, 4)


def test_stimulus_generation():
    """Test stimulus generation."""
    from hh_optimized import Stimulus
    
    # Test step stimulus
    stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
    assert len(stim) == int(100.0 / 0.01)
    assert stim[1000] == 10.0  # At t=10ms
    assert stim[5000] == 0.0   # At t=50ms
    
    # Test constant stimulus
    stim = Stimulus.constant(5.0, 100.0, 0.01)
    assert np.all(stim == 5.0)


def test_basic_simulation():
    """Test basic single neuron simulation."""
    from hh_optimized import HHModel, Simulator, Stimulus
    
    model = HHModel()
    simulator = Simulator(model=model, backend='cpu', integrator='rk4')
    
    stimulus = Stimulus.step(10.0, 10.0, 40.0, 50.0, 0.01)
    
    result = simulator.run(
        T=50.0,
        dt=0.01,
        stimulus=stimulus,
        spike_threshold=0.0
    )
    
    assert result.V is not None
    assert len(result.time) == len(result.V)
    assert result.V[0] < -60  # Starts near resting
    assert result.V.max() > 0  # Should spike
    
    spike_count = result.get_spike_count()
    assert spike_count > 0, "Should detect at least one spike"


def test_batch_simulation():
    """Test batch simulation."""
    from hh_optimized import HHModel, Simulator, Stimulus
    
    model = HHModel()
    simulator = Simulator(model=model, backend='cpu', integrator='rk4')
    
    batch_size = 5
    stimulus = Stimulus.step(10.0, 10.0, 40.0, 50.0, 0.01)
    
    result = simulator.run(
        T=50.0,
        dt=0.01,
        stimulus=stimulus,
        batch_size=batch_size,
        spike_threshold=0.0
    )
    
    assert result.V.shape[1] == batch_size
    spike_counts = result.get_spike_count()
    assert len(spike_counts) == batch_size


@pytest.mark.parametrize("integrator", ['euler', 'rk4', 'rk4rl'])
def test_different_integrators(integrator):
    """Test different integrator types."""
    from hh_optimized import HHModel, Simulator, Stimulus
    
    model = HHModel()
    stimulus = Stimulus.step(10.0, 10.0, 40.0, 50.0, 0.01)
    
    simulator = Simulator(model=model, backend='cpu', integrator=integrator)
    result = simulator.run(T=50.0, dt=0.01, stimulus=stimulus)
    assert result.V is not None


def test_numba_backend():
    """Test Numba backend if available."""
    from hh_optimized import HHModel, Simulator, Stimulus
    
    model = HHModel()
    try:
        simulator = Simulator(model=model, backend='numba', integrator='rk4')
        stimulus = Stimulus.step(10.0, 10.0, 40.0, 50.0, 0.01)
        result = simulator.run(T=50.0, dt=0.01, stimulus=stimulus)
        assert result.V is not None
    except ValueError as ve:
        if "not available" in str(ve):
            pytest.skip("Numba backend not available")
        else:
            raise

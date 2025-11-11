"""
Pytest-based validation tests for Hodgkin-Huxley implementation.

Run with: pytest test/
"""

import numpy as np
import pytest

# Try to import scipy for reference solutions
try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from cpu_backed import HHModel, Simulator, Stimulus
from hh_core.models import HHParameters, HHState, derivatives


class TestPhysiologicalBehavior:
    """Tests for physiologically correct HH behavior."""
    
    def test_resting_potential(self):
        """Test that resting potential is approximately -65 mV."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        # Run with no stimulus
        result = simulator.run(T=100.0, dt=0.01, stimulus=None)
        
        # Take average of last 10 ms (should be at rest)
        V_rest = np.mean(result.V[-1000:])
        
        # Resting potential should be around -65 mV (typically -65 to -70)
        assert -70 < V_rest < -60, \
            f"Resting potential {V_rest:.2f} mV outside expected range [-70, -60]"
    
    def test_spike_threshold(self):
        """Test that rheobase is in the expected range."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        # Test different current amplitudes
        currents = np.arange(3.0, 10.0, 0.5)
        spike_counts = []
        
        for I in currents:
            stim = Stimulus.step(I, 10.0, 200.0, 210.0, 0.01)
            result = simulator.run(T=210.0, dt=0.01, stimulus=stim)
            count = result.get_spike_count()
            spike_counts.append(count)
        
        # Find rheobase (first current that produces at least 2 spikes)
        rheobase_idx = np.where(np.array(spike_counts) >= 2)[0]
        
        assert len(rheobase_idx) > 0, "No sustained spiking detected at any current"
        
        rheobase = currents[rheobase_idx[0]]
        assert 3.0 < rheobase < 8.0, \
            f"Rheobase {rheobase:.2f} µA/cm² outside expected range [3.0, 8.0]"
    
    def test_action_potential_amplitude(self):
        """Test that action potential amplitude is physiologically realistic."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        result = simulator.run(T=100.0, dt=0.01, stimulus=stim)
        
        V_peak = result.V.max()
        V_trough = result.V.min()
        amplitude = V_peak - V_trough
        
        # Peak should be positive (overshoot)
        assert 30 < V_peak < 60, \
            f"Peak voltage {V_peak:.2f} mV outside expected range [30, 60]"
        
        # Amplitude should be ~100-120 mV
        assert 100 < amplitude < 130, \
            f"AP amplitude {amplitude:.2f} mV outside expected range [100, 130]"
    
    def test_firing_rate_increases_with_current(self):
        """Test that firing rate increases monotonically with input current (F-I curve)."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        currents = [8.0, 10.0, 12.0, 15.0, 20.0]
        firing_rates = []
        
        for I in currents:
            stim = Stimulus.constant(I, 200.0, 0.01)
            result = simulator.run(T=200.0, dt=0.01, stimulus=stim)
            
            spike_times = result.get_spike_times()
            if len(spike_times) > 1:
                rate = 1000.0 / np.mean(np.diff(spike_times))  # Hz
                firing_rates.append(rate)
            else:
                firing_rates.append(0.0)
        
        # Check monotonic increase
        for i in range(len(firing_rates) - 1):
            assert firing_rates[i] <= firing_rates[i+1], \
                f"F-I curve not monotonic: {firing_rates}"
        
        # Check reasonable range (should be < 400 Hz for HH)
        assert max(firing_rates) < 400, \
            f"Maximum firing rate {max(firing_rates):.1f} Hz exceeds physiological limit"


class TestGatingVariables:
    """Tests for gating variable behavior."""
    
    def test_gating_variables_stay_in_bounds(self):
        """Test that all gating variables stay in [0, 1]."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        stim = Stimulus.step(15.0, 10.0, 100.0, 150.0, 0.01)
        result = simulator.run(T=150.0, dt=0.01, stimulus=stim)
        
        # Check all gating variables stay in [0, 1]
        assert result.m.min() >= 0 and result.m.max() <= 1, \
            f"m out of bounds: [{result.m.min():.3f}, {result.m.max():.3f}]"
        assert result.h.min() >= 0 and result.h.max() <= 1, \
            f"h out of bounds: [{result.h.min():.3f}, {result.h.max():.3f}]"
        assert result.n.min() >= 0 and result.n.max() <= 1, \
            f"n out of bounds: [{result.n.min():.3f}, {result.n.max():.3f}]"
    
    def test_resting_gating_values(self):
        """Test that gating variables settle to correct steady-state at rest."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        result = simulator.run(T=100.0, dt=0.01, stimulus=None)
        
        # Get final values
        m_rest = result.m[-1]
        h_rest = result.h[-1]
        n_rest = result.n[-1]
        
        # Approximate steady-state values at V=-65mV
        # m should be small (sodium activation low at rest)
        assert 0.0 < m_rest < 0.1, f"m at rest = {m_rest:.3f}, expected < 0.1"
        # h should be high (sodium inactivation removed at rest)
        assert 0.5 < h_rest < 0.7, f"h at rest = {h_rest:.3f}, expected ~0.6"
        # n should be moderate (potassium activation moderate at rest)
        assert 0.2 < n_rest < 0.4, f"n at rest = {n_rest:.3f}, expected ~0.3"


class TestNumericalAccuracy:
    """Tests for numerical accuracy and stability."""
    
    def test_dt_convergence(self):
        """Test that results converge as dt decreases."""
        model = HHModel()
        
        dts = [0.05, 0.02, 0.01, 0.005]
        spike_counts = []
        max_voltages = []
        
        for dt in dts:
            simulator = Simulator(model=model, backend='cpu', integrator='rk4')
            stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, dt)
            result = simulator.run(T=100.0, dt=dt, stimulus=stim)
            
            spike_counts.append(result.get_spike_count())
            max_voltages.append(result.V.max())
        
        # Check spike counts are consistent for small dt
        assert spike_counts[-1] == spike_counts[-2], \
            f"Spike counts not converged: {spike_counts}"
        
        # Check voltage convergence (< 5% difference between smallest dts)
        voltage_diff = abs(max_voltages[-1] - max_voltages[-2]) / max_voltages[-1]
        assert voltage_diff < 0.05, \
            f"Voltage not converged: {voltage_diff*100:.2f}% difference"
    
    def test_integrator_consistency(self):
        """Test that different integrators produce consistent results."""
        model = HHModel()
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        
        # Run with RK4
        sim_rk4 = Simulator(model=model, backend='cpu', integrator='rk4')
        result_rk4 = sim_rk4.run(T=100.0, dt=0.01, stimulus=stim)
        
        # Run with RK4-RL
        sim_rk4rl = Simulator(model=model, backend='cpu', integrator='rk4rl')
        result_rk4rl = sim_rk4rl.run(T=100.0, dt=0.01, stimulus=stim)
        
        # Compare spike counts
        spike_count_rk4 = result_rk4.get_spike_count()
        spike_count_rk4rl = result_rk4rl.get_spike_count()
        
        assert spike_count_rk4 == spike_count_rk4rl, \
            f"Spike counts differ: RK4={spike_count_rk4}, RK4-RL={spike_count_rk4rl}"
        
        # Compare voltage traces (should be similar but not identical)
        voltage_rmse = np.sqrt(np.mean((result_rk4.V - result_rk4rl.V)**2))
        assert voltage_rmse < 5.0, \
            f"Voltage RMSE {voltage_rmse:.3f} mV too large between integrators"
    
    def test_no_nans_or_infs(self):
        """Test that simulation doesn't produce NaN or Inf values."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        result = simulator.run(T=100.0, dt=0.01, stimulus=stim)
        
        assert not np.any(np.isnan(result.V)), "Voltage contains NaN"
        assert not np.any(np.isinf(result.V)), "Voltage contains Inf"
        assert not np.any(np.isnan(result.m)), "m contains NaN"
        assert not np.any(np.isnan(result.h)), "h contains NaN"
        assert not np.any(np.isnan(result.n)), "n contains NaN"


class TestBatchSimulation:
    """Tests for batch simulation functionality."""
    
    def test_batch_single_equivalence(self):
        """Test that batch of 1 gives same result as single simulation."""
        model = HHModel()
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        
        # Single neuron
        sim_single = Simulator(model=model, backend='cpu', integrator='rk4')
        result_single = sim_single.run(T=100.0, dt=0.01, stimulus=stim, batch_size=1)
        
        # Batch of 1
        sim_batch = Simulator(model=model, backend='cpu', integrator='rk4')
        result_batch = sim_batch.run(T=100.0, dt=0.01, stimulus=stim, batch_size=1)
        
        # Should give identical results
        assert np.allclose(result_single.V, result_batch.V), \
            "Single and batch(1) produce different voltages"
    
    def test_batch_independence(self):
        """Test that neurons in batch are independent (same stimulus -> same output)."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        batch_size = 5
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        
        result = simulator.run(T=100.0, dt=0.01, stimulus=stim, batch_size=batch_size)
        
        # All neurons get same stimulus, so should produce identical traces
        for i in range(1, batch_size):
            assert np.allclose(result.V[:, 0], result.V[:, i]), \
                f"Neuron {i} differs from neuron 0 (should be identical)"


@pytest.mark.skipif(not SCIPY_AVAILABLE, reason="scipy not installed")
class TestAgainstSciPy:
    """Validation against scipy reference implementation."""
    
    def test_scipy_reference_comparison(self):
        """Compare our implementation against scipy's ODE solver."""
        params = HHParameters()
        state0 = HHState.resting_state(1)
        
        # Define ODE for scipy
        def ode_func(t, y):
            state = HHState(y)
            I_ext = 10.0 if 10.0 <= t <= 40.0 else 0.0
            return derivatives(state, I_ext, params)
        
        # Solve with scipy
        t_span = (0, 100.0)
        t_eval = np.linspace(0, 100.0, 10001)
        sol_scipy = solve_ivp(
            ode_func, t_span, state0.data, 
            method='RK45', t_eval=t_eval, 
            rtol=1e-6, atol=1e-8
        )
        
        # Solve with our implementation
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        result_ours = simulator.run(T=100.0, dt=0.01, stimulus=stim)
        
        # Compare voltages
        V_scipy = sol_scipy.y[0, :]
        V_ours = result_ours.V
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((V_scipy - V_ours)**2))
        
        # Should be quite close (< 2 mV RMSE)
        assert rmse < 2.0, \
            f"RMSE vs scipy {rmse:.4f} mV exceeds threshold"


class TestStimulusGeneration:
    """Tests for stimulus generation utilities."""
    
    def test_step_stimulus(self):
        """Test step stimulus generation."""
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        
        assert len(stim) == 10000
        assert stim[500] == 0.0  # Before step
        assert stim[1500] == 10.0  # During step
        assert stim[5000] == 0.0  # After step
    
    def test_constant_stimulus(self):
        """Test constant stimulus generation."""
        stim = Stimulus.constant(5.0, 100.0, 0.01)
        
        assert len(stim) == 10000
        assert np.all(stim == 5.0)
    
    def test_pulse_train_stimulus(self):
        """Test pulse train generation."""
        stim = Stimulus.pulse_train(
            amplitude=10.0,
            pulse_duration=2.0,
            pulse_period=10.0,
            n_pulses=3,
            t_start=10.0,
            duration=100.0,
            dt=0.01
        )
        
        assert len(stim) == 10000
        # Should have pulses at t=10, 20, 30 ms
        assert stim[1100] == 10.0  # First pulse
        assert stim[2100] == 10.0  # Second pulse
        assert stim[3100] == 10.0  # Third pulse

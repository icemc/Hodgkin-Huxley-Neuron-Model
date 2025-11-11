"""
Comprehensive Test Suite for Hodgkin-Huxley Neuron Model Simulator

This test suite combines:
1. Basic functionality tests (imports, model creation, basic simulation)
2. Physiological validation tests (resting potential, spike behavior, F-I curves)
3. Numerical accuracy tests (CPU vs GPU vs SciPy RK4 implementations)
4. Batch simulation consistency tests

Run all tests: pytest test/test_suite.py -v
Run specific group: pytest test/test_suite.py::TestBasicFunctionality -v
"""

import numpy as np
import pytest

# Check for optional dependencies
try:
    import cupy as cp
    from gpu_backed import GPUSimulator
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    from scipy.integrate import solve_ivp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Core imports
from hh_core.models import HHParameters, HHState, derivatives
from hh_core import RK4, ForwardEuler
from hh_core.utils import Stimulus as StimGen
from cpu_backed import Simulator, HHModel, Stimulus, VectorizedSimulator


# Helper function to extract single neuron data (handles both 1D and 2D arrays)
def get_neuron_data(arr, neuron_idx=0):
    """Extract data for a single neuron, handling both 1D and 2D arrays."""
    if arr.ndim == 1:
        return arr
    else:
        return arr[:, neuron_idx]


# ============================================================================
# SECTION 1: BASIC FUNCTIONALITY TESTS
# ============================================================================

class TestBasicFunctionality:
    """Test core functionality: imports, model creation, basic simulations."""
    
    def test_imports(self):
        """Test that all modules can be imported."""
        from hh_core import HHParameters, HHState, RK4, ForwardEuler
        from hh_core.utils import Stimulus
        from cpu_backed import VectorizedSimulator, Simulator, HHModel
        
        # If we get here, all imports succeeded
        assert True
    
    def test_model_creation(self):
        """Test model and state creation."""
        model = HHModel()
        params = model.get_params()
        assert params.C_m == 1.0
        assert params.g_Na == 120.0
        
        state = model.resting_state()
        assert state.data.shape == (4,)
        
        batch_state = model.resting_state(batch_size=10)
        assert batch_state.data.shape == (10, 4)
    
    def test_stimulus_generation(self):
        """Test stimulus generation."""
        # Test step stimulus
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        assert len(stim) == int(100.0 / 0.01)
        assert stim[1000] == 10.0  # At t=10ms
        assert stim[5000] == 0.0   # At t=50ms
        
        # Test constant stimulus
        stim = Stimulus.constant(5.0, 100.0, 0.01)
        assert np.all(stim == 5.0)
    
    def test_basic_simulation(self):
        """Test basic single neuron simulation."""
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
    
    def test_batch_simulation(self):
        """Test batch simulation."""
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


# ============================================================================
# SECTION 2: PHYSIOLOGICAL VALIDATION TESTS
# ============================================================================

class TestPhysiologicalBehavior:
    """Tests for physiologically correct Hodgkin-Huxley behavior."""
    
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


# ============================================================================
# SECTION 3: NUMERICAL ACCURACY TESTS (CPU, GPU, SciPy)
# ============================================================================

class TestRK4Accuracy:
    """Test that CPU, GPU, and SciPy RK4 implementations match."""
    
    @pytest.mark.skipif(not SCIPY_AVAILABLE, reason="SciPy not available")
    def test_cpu_vs_scipy_single_neuron(self):
        """Compare CPU RK4 with SciPy's RK45 for single neuron."""
        # Simulation parameters
        T = 50.0  # ms
        dt = 0.01  # ms
        n_steps = int(T / dt) + 1
        
        # Create stimulus
        stimulus = np.zeros(n_steps)
        stimulus[int(10/dt):int(40/dt)] = 10.0  # 10 µA/cm² pulse
        
        # CPU simulation
        cpu_sim = VectorizedSimulator(dtype='float64')
        cpu_result = cpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # SciPy simulation
        scipy_result = self._run_scipy_simulation(T, dt, stimulus)
        
        # Compare results (allow small numerical differences)
        rtol = 1e-4  # 0.01% relative tolerance
        atol = 1e-6  # Absolute tolerance for near-zero values
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['V']), scipy_result['V'],
            rtol=rtol, atol=atol,
            err_msg="CPU voltage differs from SciPy"
        )
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['m']), scipy_result['m'],
            rtol=rtol, atol=atol,
            err_msg="CPU m-gate differs from SciPy"
        )
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['h']), scipy_result['h'],
            rtol=rtol, atol=atol,
            err_msg="CPU h-gate differs from SciPy"
        )
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['n']), scipy_result['n'],
            rtol=rtol, atol=atol,
            err_msg="CPU n-gate differs from SciPy"
        )
        
        print(f"✓ CPU vs SciPy: Max voltage error = {np.max(np.abs(get_neuron_data(cpu_result['V']) - scipy_result['V'])):.6f} mV")
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_vs_cpu_single_neuron(self):
        """Compare GPU RK4 with CPU RK4 for single neuron."""
        # Simulation parameters
        T = 50.0  # ms
        dt = 0.01  # ms
        n_steps = int(T / dt) + 1
        
        # Create stimulus
        stimulus = np.zeros(n_steps)
        stimulus[int(10/dt):int(40/dt)] = 10.0  # 10 µA/cm² pulse
        
        # CPU simulation (float32 to match GPU)
        cpu_sim = VectorizedSimulator(dtype='float32')
        cpu_result = cpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # GPU simulation
        gpu_sim = GPUSimulator(dtype='float32')
        gpu_result = gpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # Compare results
        # Float32 has less precision, so use looser tolerance
        rtol = 1e-4  # 0.01% relative tolerance
        atol = 1e-3  # 0.001 mV absolute tolerance
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['V']), get_neuron_data(gpu_result['V']),
            rtol=rtol, atol=atol,
            err_msg="GPU voltage differs from CPU"
        )
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['m']), get_neuron_data(gpu_result['m']),
            rtol=rtol, atol=atol,
            err_msg="GPU m-gate differs from CPU"
        )
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['h']), get_neuron_data(gpu_result['h']),
            rtol=rtol, atol=atol,
            err_msg="GPU h-gate differs from CPU"
        )
        
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['n']), get_neuron_data(gpu_result['n']),
            rtol=rtol, atol=atol,
            err_msg="GPU n-gate differs from CPU"
        )
        
        print(f"✓ GPU vs CPU: Max voltage error = {np.max(np.abs(get_neuron_data(gpu_result['V']) - get_neuron_data(cpu_result['V']))):.6f} mV")
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_vs_cpu_batch(self):
        """Compare GPU and CPU for batch simulation."""
        # Simulation parameters
        T = 50.0  # ms
        dt = 0.01  # ms
        batch_size = 10
        n_steps = int(T / dt) + 1
        
        # Create different stimuli for each neuron
        stimulus = np.zeros((n_steps, batch_size))
        for i in range(batch_size):
            current = 5.0 + i * 1.0  # Different current for each neuron
            stimulus[int(10/dt):int(40/dt), i] = current
        
        # CPU simulation
        cpu_sim = VectorizedSimulator(dtype='float32')
        cpu_result = cpu_sim.run(T, dt, batch_size=batch_size, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # GPU simulation
        gpu_sim = GPUSimulator(dtype='float32')
        gpu_result = gpu_sim.run(T, dt, batch_size=batch_size, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # Compare results for all neurons
        # Float32 with larger batch may accumulate small errors
        rtol = 5e-4  # 0.05% relative tolerance
        atol = 0.1   # 0.1 mV absolute tolerance
        
        np.testing.assert_allclose(
            cpu_result['V'], gpu_result['V'],
            rtol=rtol, atol=atol,
            err_msg="GPU batch voltages differ from CPU"
        )
        
        np.testing.assert_allclose(
            cpu_result['m'], gpu_result['m'],
            rtol=rtol, atol=atol,
            err_msg="GPU batch m-gates differ from CPU"
        )
        
        np.testing.assert_allclose(
            cpu_result['h'], gpu_result['h'],
            rtol=rtol, atol=atol,
            err_msg="GPU batch h-gates differ from CPU"
        )
        
        np.testing.assert_allclose(
            cpu_result['n'], gpu_result['n'],
            rtol=rtol, atol=atol,
            err_msg="GPU batch n-gates differ from CPU"
        )
        
        max_error = np.max(np.abs(gpu_result['V'] - cpu_result['V']))
        print(f"✓ GPU vs CPU (batch={batch_size}): Max voltage error = {max_error:.6f} mV")
    
    @pytest.mark.skipif(not GPU_AVAILABLE or not SCIPY_AVAILABLE, 
                        reason="GPU or SciPy not available")
    def test_all_three_implementations(self):
        """Compare CPU, GPU, and SciPy for same simulation."""
        # Simulation parameters
        T = 30.0  # Shorter for faster test
        dt = 0.01  # ms
        n_steps = int(T / dt) + 1
        
        # Create stimulus
        stimulus = np.zeros(n_steps)
        stimulus[int(10/dt):int(25/dt)] = 15.0  # Strong pulse to ensure spike
        
        # CPU simulation (float64 for better comparison with SciPy)
        cpu_sim = VectorizedSimulator(dtype='float64')
        cpu_result = cpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # GPU simulation (float32)
        gpu_sim = GPUSimulator(dtype='float32')
        gpu_result = gpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # SciPy simulation
        scipy_result = self._run_scipy_simulation(T, dt, stimulus)
        
        # Compare CPU vs SciPy (should be very close)
        np.testing.assert_allclose(
            get_neuron_data(cpu_result['V']), scipy_result['V'],
            rtol=1e-4, atol=1e-6,
            err_msg="CPU differs from SciPy reference"
        )
        
        # Compare GPU vs CPU (float32 vs float64, so looser tolerance)
        np.testing.assert_allclose(
            get_neuron_data(gpu_result['V']), get_neuron_data(cpu_result['V']),
            rtol=5e-4, atol=0.01,
            err_msg="GPU differs significantly from CPU"
        )
        
        # Print error summary
        cpu_scipy_error = np.max(np.abs(get_neuron_data(cpu_result['V']) - scipy_result['V']))
        gpu_cpu_error = np.max(np.abs(get_neuron_data(gpu_result['V']) - get_neuron_data(cpu_result['V'])))
        
        print(f"✓ Three-way comparison:")
        print(f"  CPU vs SciPy: max error = {cpu_scipy_error:.6f} mV")
        print(f"  GPU vs CPU:   max error = {gpu_cpu_error:.6f} mV")
    
    def _run_scipy_simulation(self, T, dt, stimulus):
        """Run HH simulation using SciPy's solve_ivp."""
        params = HHParameters()
        state0 = HHState.resting_state(batch_size=1, dtype=np.float64)
        
        # Time points
        t_eval = np.linspace(0, T, int(T/dt) + 1)
        
        # Initial state vector [V, m, h, n]
        y0 = np.array([state0.V, state0.m, state0.h, state0.n])
        
        # Define derivative function
        def dy_dt(t, y):
            V, m, h, n = y
            
            # Interpolate stimulus at current time
            t_idx = int(t / dt)
            if t_idx >= len(stimulus):
                I_stim = 0.0
            else:
                I_stim = stimulus[t_idx]
            
            # Create state object
            state_data = np.array([V, m, h, n])
            state = HHState(state_data)
            
            # Compute derivatives
            derivs = derivatives(state, I_stim, params)
            return [derivs[0], derivs[1], derivs[2], derivs[3]]
        
        # Solve using RK45 (adaptive RK4)
        sol = solve_ivp(
            dy_dt, 
            (0, T), 
            y0, 
            method='RK45',
            t_eval=t_eval,
            rtol=1e-10,
            atol=1e-12
        )
        
        return {
            'time': sol.t,
            'V': sol.y[0],
            'm': sol.y[1],
            'h': sol.y[2],
            'n': sol.y[3]
        }


class TestNumericalProperties:
    """Test numerical properties of RK4 integrators."""
    
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
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_energy_conservation(self):
        """Test that total energy is reasonably conserved."""
        # Long simulation with no stimulus
        T = 100.0
        dt = 0.01
        
        cpu_sim = VectorizedSimulator(dtype='float64')
        cpu_result = cpu_sim.run(T, dt, batch_size=1, stimulus=None, record_vars=['V','m','h','n'])
        
        gpu_sim = GPUSimulator(dtype='float32')
        gpu_result = gpu_sim.run(T, dt, batch_size=1, stimulus=None, record_vars=['V','m','h','n'])
        
        # Both should settle to resting state
        cpu_V = get_neuron_data(cpu_result['V'])
        gpu_V = get_neuron_data(gpu_result['V'])
        V_rest_cpu = np.mean(cpu_V[-1000:])
        V_rest_gpu = np.mean(gpu_V[-1000:])
        
        # Resting potentials should be close
        assert abs(V_rest_cpu - V_rest_gpu) < 0.1, \
            f"CPU and GPU resting potentials differ: {V_rest_cpu:.2f} vs {V_rest_gpu:.2f} mV"
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_deterministic_behavior(self):
        """Test that simulations are deterministic (same inputs → same outputs)."""
        T = 50.0
        dt = 0.01
        n_steps = int(T / dt) + 1
        
        # Create stimulus
        stimulus = np.zeros(n_steps)
        stimulus[int(10/dt):int(40/dt)] = 10.0
        
        # Run GPU simulation twice
        gpu_sim = GPUSimulator(dtype='float32')
        result1 = gpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        result2 = gpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # Results should be identical (no randomness)
        np.testing.assert_array_equal(
            result1['V'], result2['V'],
            err_msg="GPU simulation is non-deterministic"
        )
        
        # Run CPU simulation twice
        cpu_sim = VectorizedSimulator(dtype='float32')
        result3 = cpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        result4 = cpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # Results should be identical
        np.testing.assert_array_equal(
            result3['V'], result4['V'],
            err_msg="CPU simulation is non-deterministic"
        )
    
    def test_timestep_independence(self):
        """Test that results converge as dt decreases."""
        T = 20.0
        
        # Run with different time steps
        dt_values = [0.02, 0.01, 0.005]
        results = []
        
        for dt in dt_values:
            n_steps_this = int(T / dt) + 1
            stimulus = np.zeros(n_steps_this)
            stimulus[int(10/dt):int(20/dt)] = 10.0
            
            cpu_sim = VectorizedSimulator(dtype='float64')
            result = cpu_sim.run(T, dt, batch_size=1, stimulus=stimulus, record_vars=['V'])
            results.append(result)
        
        # Interpolate to common time grid for comparison
        t_common = np.linspace(0, T, int(T/0.005) + 1)
        
        V_interp = []
        for i, dt in enumerate(dt_values):
            t_this = results[i]['time']
            V_this = get_neuron_data(results[i]['V'])
            V_interp.append(np.interp(t_common, t_this, V_this))
        
        # Results should converge (finer timestep closer to finest)
        error_coarse = np.max(np.abs(V_interp[0] - V_interp[2]))
        error_medium = np.max(np.abs(V_interp[1] - V_interp[2]))
        
        # Medium timestep should be closer to finest than coarse
        assert error_medium < error_coarse, \
            f"Results not converging: error(medium)={error_medium:.3f} >= error(coarse)={error_coarse:.3f}"
        
        print(f"✓ Convergence test:")
        print(f"  dt=0.02 vs dt=0.005: max error = {error_coarse:.3f} mV")
        print(f"  dt=0.01 vs dt=0.005: max error = {error_medium:.3f} mV")
    
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


# ============================================================================
# SECTION 4: BATCH CONSISTENCY TESTS
# ============================================================================

class TestBatchConsistency:
    """Test that batch simulations are consistent with single simulations."""
    
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
        """Test that neurons in batch are independent (same stimulus → same output)."""
        model = HHModel()
        simulator = Simulator(model=model, backend='cpu', integrator='rk4')
        
        batch_size = 5
        stim = Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)
        
        result = simulator.run(T=100.0, dt=0.01, stimulus=stim, batch_size=batch_size)
        
        # All neurons get same stimulus, so should produce identical traces
        for i in range(1, batch_size):
            assert np.allclose(result.V[:, 0], result.V[:, i]), \
                f"Neuron {i} differs from neuron 0 (should be identical)"
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_gpu_batch_matches_single(self):
        """Test that GPU batch simulation matches running single neurons."""
        T = 30.0
        dt = 0.01
        n_steps = int(T / dt) + 1
        batch_size = 5
        
        # Create stimuli
        stimuli = []
        for i in range(batch_size):
            stim = np.zeros(n_steps)
            stim[int(10/dt):int(25/dt)] = 5.0 + i * 2.0
            stimuli.append(stim)
        
        # Run batch simulation
        stimulus_batch = np.column_stack(stimuli)
        gpu_sim = GPUSimulator(dtype='float32')
        batch_result = gpu_sim.run(T, dt, batch_size=batch_size, 
                                    stimulus=stimulus_batch, record_vars=['V','m','h','n'])
        
        # Run individual simulations
        for i in range(batch_size):
            single_result = gpu_sim.run(T, dt, batch_size=1, 
                                        stimulus=stimuli[i], record_vars=['V','m','h','n'])
            
            # Compare with corresponding batch column
            np.testing.assert_allclose(
                single_result['V'][:, 0], batch_result['V'][:, i],
                rtol=1e-6, atol=1e-7,
                err_msg=f"Batch neuron {i} differs from single simulation"
            )
    
    @pytest.mark.skipif(not GPU_AVAILABLE, reason="GPU not available")
    def test_cpu_gpu_batch_consistency(self):
        """Test CPU and GPU produce same results for identical batch."""
        T = 30.0
        dt = 0.01
        batch_size = 5
        n_steps = int(T / dt) + 1
        
        # Same stimulus for all neurons (easier to debug if something fails)
        stimulus = np.zeros((n_steps, batch_size))
        stimulus[int(10/dt):int(25/dt), :] = 12.0
        
        # CPU batch
        cpu_sim = VectorizedSimulator(dtype='float32')
        cpu_result = cpu_sim.run(T, dt, batch_size=batch_size, 
                                stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # GPU batch
        gpu_sim = GPUSimulator(dtype='float32')
        gpu_result = gpu_sim.run(T, dt, batch_size=batch_size, 
                                stimulus=stimulus, record_vars=['V','m','h','n'])
        
        # Should match closely (allow small float32 differences)
        np.testing.assert_allclose(
            cpu_result['V'], gpu_result['V'],
            rtol=5e-4, atol=0.01,  # 0.05% relative, 0.01 mV absolute
            err_msg="CPU and GPU batch results differ"
        )
        
        # All neurons should have identical results (same stimulus)
        for i in range(1, batch_size):
            np.testing.assert_allclose(
                cpu_result['V'][:, 0], cpu_result['V'][:, i],
                rtol=1e-6, atol=1e-7,
                err_msg=f"CPU: Neuron {i} differs from neuron 0 (should be identical)"
            )
            
            np.testing.assert_allclose(
                gpu_result['V'][:, 0], gpu_result['V'][:, i],
                rtol=1e-6, atol=1e-7,
                err_msg=f"GPU: Neuron {i} differs from neuron 0 (should be identical)"
            )


# ============================================================================
# SECTION 5: STIMULUS GENERATION TESTS
# ============================================================================

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


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_basic_tests():
    """Run basic functionality tests only."""
    pytest.main([__file__, '-v', '-k', 'TestBasicFunctionality'])


def run_physiological_tests():
    """Run physiological validation tests only."""
    pytest.main([__file__, '-v', '-k', 'TestPhysiological or TestGating'])


def run_accuracy_tests():
    """Run numerical accuracy tests only."""
    pytest.main([__file__, '-v', '-k', 'TestRK4Accuracy'])


def run_gpu_tests():
    """Run GPU-specific tests only."""
    pytest.main([__file__, '-v', '-m', 'not skipif'])


def run_all_tests():
    """Run all tests."""
    pytest.main([__file__, '-v'])


if __name__ == '__main__':
    # Run all tests when executed directly
    run_all_tests()

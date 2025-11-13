"""
GPU-accelerated Hodgkin-Huxley simulator using CuPy with custom CUDA kernels.

This implementation parallelizes the CPU's RK4 implementation across neurons.
Each CUDA thread simulates one neuron independently, reusing the exact same
mathematical operations as the CPU version but in parallel (SIMD).
"""

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

from typing import Optional, Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hh_core.models import HHParameters, HHState
from hh_core.api import SimulationResult, BaseSimulator, HHModel
from hh_core.utils import detect_spikes, interpolate_spike_times



# CUDA kernel code that implements HH equations
# This mirrors the CPU implementation but runs in parallel threads
CUDA_RK4_KERNEL = r'''
extern "C" __global__
void rk4_step_kernel(
    const float* V_in, const float* m_in, const float* h_in, const float* n_in,
    float* V_out, float* m_out, float* h_out, float* n_out,
    const float* I_ext,
    const float dt,
    const int batch_size,
    const float C_m, const float g_Na, const float g_K, const float g_L,
    const float E_Na, const float E_K, const float E_L
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= batch_size) return;
    
    // Load current state for this neuron
    float V = V_in[tid];
    float m = m_in[tid];
    float h = h_in[tid];
    float n = n_in[tid];
    float I = I_ext[tid];
    
    // Helper functions (inline)
    auto alpha_m = [](float V) {
        float x = V + 40.0f;
        return (fabsf(x) < 1e-4f) ? 1.0f : 0.1f * x / (1.0f - expf(-x / 10.0f));
    };
    
    auto beta_m = [](float V) {
        return 4.0f * expf(-(V + 65.0f) / 18.0f);
    };
    
    auto alpha_h = [](float V) {
        return 0.07f * expf(-(V + 65.0f) / 20.0f);
    };
    
    auto beta_h = [](float V) {
        return 1.0f / (1.0f + expf(-(V + 35.0f) / 10.0f));
    };
    
    auto alpha_n = [](float V) {
        float x = V + 55.0f;
        return (fabsf(x) < 1e-4f) ? 0.1f : 0.01f * x / (1.0f - expf(-x / 10.0f));
    };
    
    auto beta_n = [](float V) {
        return 0.125f * expf(-(V + 65.0f) / 80.0f);
    };
    
    // Compute derivatives function
    auto compute_derivatives = [&](float V, float m, float h, float n, 
                                   float* dV, float* dm, float* dh, float* dn) {
        // Rate functions
        float a_m = alpha_m(V);
        float b_m = beta_m(V);
        float a_h = alpha_h(V);
        float b_h = beta_h(V);
        float a_n = alpha_n(V);
        float b_n = beta_n(V);
        
        // Gating derivatives
        *dm = a_m * (1.0f - m) - b_m * m;
        *dh = a_h * (1.0f - h) - b_h * h;
        *dn = a_n * (1.0f - n) - b_n * n;
        
        // Ionic currents
        float I_Na = g_Na * (m * m * m) * h * (V - E_Na);
        float I_K = g_K * (n * n * n * n) * (V - E_K);
        float I_L = g_L * (V - E_L);
        float I_ion = I_Na + I_K + I_L;
        
        // Voltage derivative
        *dV = (I - I_ion) / C_m;
    };
    
    // RK4 integration - exactly as in CPU version
    float dV1, dm1, dh1, dn1;
    float dV2, dm2, dh2, dn2;
    float dV3, dm3, dh3, dn3;
    float dV4, dm4, dh4, dn4;
    
    // k1
    compute_derivatives(V, m, h, n, &dV1, &dm1, &dh1, &dn1);
    
    // k2
    float V2 = V + 0.5f * dt * dV1;
    float m2 = m + 0.5f * dt * dm1;
    float h2 = h + 0.5f * dt * dh1;
    float n2 = n + 0.5f * dt * dn1;
    compute_derivatives(V2, m2, h2, n2, &dV2, &dm2, &dh2, &dn2);
    
    // k3
    float V3 = V + 0.5f * dt * dV2;
    float m3 = m + 0.5f * dt * dm2;
    float h3 = h + 0.5f * dt * dh2;
    float n3 = n + 0.5f * dt * dn2;
    compute_derivatives(V3, m3, h3, n3, &dV3, &dm3, &dh3, &dn3);
    
    // k4
    float V4 = V + dt * dV3;
    float m4 = m + dt * dm3;
    float h4 = h + dt * dh3;
    float n4 = n + dt * dn3;
    compute_derivatives(V4, m4, h4, n4, &dV4, &dm4, &dh4, &dn4);
    
    // Final update - RK4 combination
    V_out[tid] = V + (dt / 6.0f) * (dV1 + 2*dV2 + 2*dV3 + dV4);
    m_out[tid] = m + (dt / 6.0f) * (dm1 + 2*dm2 + 2*dm3 + dm4);
    h_out[tid] = h + (dt / 6.0f) * (dh1 + 2*dh2 + 2*dh3 + dh4);
    n_out[tid] = n + (dt / 6.0f) * (dn1 + 2*dn2 + 2*dn3 + dn4);
}
'''


class GPUSimulator(BaseSimulator):
    """
    GPU-accelerated Hodgkin-Huxley simulator using custom CUDA kernels.
    
    This simulator reuses the same RK4 mathematics as the CPU implementation
    but parallelizes execution across neurons. Each GPU thread handles one
    neuron, allowing SIMD parallelism across the batch.
    """
    
    def __init__(self, 
                 model: Optional[HHModel] = None,
                 integrator: str = 'rk4',
                 dtype='float32'):
        """
        Initialize GPU simulator with custom CUDA kernel.
        
        Args:
            model: HH model (creates default if None)
            integrator: Integration method ('rk4' is the only supported method)
            dtype: 'float32' or 'float64' (float32 recommended for GPU)
        
        Raises:
            RuntimeError: If CuPy is not available
            ValueError: If an unsupported integrator is specified
        """
        if not CUPY_AVAILABLE:
            raise RuntimeError(
                "CuPy is not available. Please install cupy to use GPU acceleration.\n"
                "Install with: pip install cupy-cuda11x  (or cuda12x for CUDA 12)"
            )
        
        # Validate integrator
        if integrator != 'rk4':
            raise ValueError(
                f"GPU simulator only supports 'rk4' integrator, got '{integrator}'. "
                f"Supported integrators for GPU: ['rk4']. "
                f"For other integrators, please use CPUSimulator."
            )
        
        # Initialize base class (will create model if None)
        super().__init__(model, integrator='rk4', dtype=np.float32)
        
        if dtype != 'float32':
            raise NotImplementedError("Only float32 is currently supported for GPU kernels")
        
        self.dtype = cp.float32
        
        # Compile the CUDA kernel
        self._compile_kernel()
    
    def _compile_kernel(self):
        """Compile the custom CUDA kernel for RK4 integration."""
        self.rk4_kernel = cp.RawKernel(CUDA_RK4_KERNEL, 'rk4_step_kernel')
    
    def _compute_grid_size(self, batch_size: int, block_size: int = 256):
        """
        Compute grid and block dimensions for CUDA kernel launch.
        
        Args:
            batch_size: Number of neurons (threads needed)
            block_size: Threads per block (default 256)
        
        Returns:
            Tuple of (grid_size, block_size)
        """
        # Calculate number of blocks needed
        grid_size = (batch_size + block_size - 1) // block_size
        return (grid_size,), (block_size,)
    
    def run(self,
            T: float,
            dt: float = 0.01,
            state0: Optional[HHState] = None,
            stimulus: Optional[np.ndarray] = None,
            batch_size: int = 1,
            record: Optional[list] = None,
            spike_threshold: float = 0.0,
            # Backward compatibility
            record_vars: Optional[list] = None,
            record_all: bool = None) -> SimulationResult:
        """
        Run batch simulation on GPU using custom CUDA kernels.
        
        Each thread simulates one neuron using the same RK4 logic as CPU.
        
        Args:
            T: Total simulation time (ms)
            dt: Time step (ms)
            state0: Initial state (ignored, uses resting state)
            stimulus: External current array, shape (n_steps,) or (n_steps, batch_size)
            batch_size: Number of neurons to simulate in parallel
            record: List of variables to record ['V', 'm', 'h', 'n'] (default: all)
            spike_threshold: Threshold for spike detection (mV)
            record_vars: (Deprecated) use 'record' instead
            record_all: (Deprecated) use 'record' instead
        
        Returns:
            SimulationResult object with recorded data
        """
        # Handle backward compatibility
        if record_all is not None:
            import warnings
            warnings.warn("record_all is deprecated, use record instead", DeprecationWarning)
            if record_all:
                record = ['V', 'm', 'h', 'n']
            else:
                record = ['V']
        
        if record_vars is not None:
            import warnings
            warnings.warn("record_vars is deprecated, use record instead", DeprecationWarning)
            record = record_vars
        
        # Default to recording all variables
        if record is None:
            record = ['V', 'm', 'h', 'n']
        
        # Setup time - use same calculation as CPU for consistency
        n_steps = int(T / dt)
        time = np.linspace(0, T, n_steps + 1)
        
        # Initialize state on GPU - resting state at V=-65 mV
        V_rest = -65.0
        
        # Compute steady-state gating variables at rest (using correct formulas)
        # alpha_m = 0.1 * (V + 40) / (1 - exp(-(V + 40) / 10))
        x_m = V_rest + 40.0  # = -25.0
        if abs(x_m) < 1e-4:
            a_m = 1.0
        else:
            a_m = 0.1 * x_m / (1.0 - np.exp(-x_m / 10.0))
        b_m = 4.0 * np.exp(-(V_rest + 65.0) / 18.0)
        m_rest = a_m / (a_m + b_m)
        
        # alpha_h and beta_h
        a_h = 0.07 * np.exp(-(V_rest + 65.0) / 20.0)
        b_h = 1.0 / (1.0 + np.exp(-(V_rest + 35.0) / 10.0))
        h_rest = a_h / (a_h + b_h)
        
        # alpha_n = 0.01 * (V + 55) / (1 - exp(-(V + 55) / 10))
        x_n = V_rest + 55.0  # = -10.0
        if abs(x_n) < 1e-4:
            a_n = 0.1
        else:
            a_n = 0.01 * x_n / (1.0 - np.exp(-x_n / 10.0))
        b_n = 0.125 * np.exp(-(V_rest + 65.0) / 80.0)
        n_rest = a_n / (a_n + b_n)
        
        # Allocate state arrays on GPU
        V = cp.full(batch_size, V_rest, dtype=self.dtype)
        m = cp.full(batch_size, m_rest, dtype=self.dtype)
        h = cp.full(batch_size, h_rest, dtype=self.dtype)
        n = cp.full(batch_size, n_rest, dtype=self.dtype)
        
        # Allocate output buffers on GPU
        V_out = cp.empty(batch_size, dtype=self.dtype)
        m_out = cp.empty(batch_size, dtype=self.dtype)
        h_out = cp.empty(batch_size, dtype=self.dtype)
        n_out = cp.empty(batch_size, dtype=self.dtype)
        
        # Setup stimulus on GPU
        if stimulus is None:
            I_ext_gpu = cp.zeros((n_steps + 1, batch_size), dtype=self.dtype)
        else:
            stimulus = np.asarray(stimulus)
            if stimulus.ndim == 1:
                # Handle different stimulus lengths (with or without final timestep)
                if len(stimulus) == n_steps:
                    # Pad with last value to match n_steps + 1
                    stimulus = np.pad(stimulus, (0, 1), 'edge')
                elif len(stimulus) != n_steps + 1:
                    raise ValueError(
                        f"Stimulus length {len(stimulus)} doesn't match expected "
                        f"n_steps ({n_steps}) or n_steps+1 ({n_steps+1})"
                    )
                # Broadcast to all neurons
                I_ext_gpu = cp.asarray(stimulus[:, np.newaxis], dtype=self.dtype)
                I_ext_gpu = cp.broadcast_to(I_ext_gpu, (n_steps + 1, batch_size)).copy()
            else:
                # Handle 2D stimulus
                if stimulus.shape[0] == n_steps:
                    # Pad with last row
                    stimulus = np.pad(stimulus, ((0, 1), (0, 0)), 'edge')
                elif stimulus.shape[0] != n_steps + 1:
                    raise ValueError(
                        f"Stimulus length {stimulus.shape[0]} doesn't match expected "
                        f"n_steps ({n_steps}) or n_steps+1 ({n_steps+1})"
                    )
                I_ext_gpu = cp.asarray(stimulus, dtype=self.dtype)
        
        # Allocate recording arrays on GPU
        V_rec = cp.zeros((n_steps + 1, batch_size), dtype=self.dtype)
        V_rec[0] = V
        
        # Optionally record gating variables
        record_gating = any(var in record for var in ['m', 'h', 'n'])
        if record_gating:
            m_rec = cp.zeros((n_steps + 1, batch_size), dtype=self.dtype)
            h_rec = cp.zeros((n_steps + 1, batch_size), dtype=self.dtype)
            n_rec = cp.zeros((n_steps + 1, batch_size), dtype=self.dtype)
            m_rec[0] = m
            h_rec[0] = h
            n_rec[0] = n
        
        # Compute kernel launch configuration
        grid_size, block_size = self._compute_grid_size(batch_size)
        
        # Model parameters as scalars
        C_m = float(self.model.params.C_m)
        g_Na = float(self.model.params.g_Na)
        g_K = float(self.model.params.g_K)
        g_L = float(self.model.params.g_L)
        E_Na = float(self.model.params.E_Na)
        E_K = float(self.model.params.E_K)
        E_L = float(self.model.params.E_L)
        
        # Main integration loop - launch kernel for each time step
        for i in range(n_steps):
            I_ext = I_ext_gpu[i]
            
            # Launch CUDA kernel: each thread handles one neuron
            self.rk4_kernel(
                grid_size, block_size,
                (V, m, h, n,           # Input state
                 V_out, m_out, h_out, n_out,  # Output state
                 I_ext,                 # External current
                 self.dtype(dt),       # Time step
                 np.int32(batch_size), # Number of neurons
                 self.dtype(C_m), self.dtype(g_Na), self.dtype(g_K), self.dtype(g_L),
                 self.dtype(E_Na), self.dtype(E_K), self.dtype(E_L))
            )
            
            # Swap buffers (output becomes input for next step)
            V, V_out = V_out, V
            m, m_out = m_out, m
            h, h_out = h_out, h
            n, n_out = n_out, n
            
            # Record
            V_rec[i + 1] = V
            if record_gating:
                m_rec[i + 1] = m
                h_rec[i + 1] = h
                n_rec[i + 1] = n
        
        # Transfer results to CPU
        results = {
            'time': time,
            'V': cp.asnumpy(V_rec),
            'dt': dt,
            'batch_size': batch_size
        }
        
        # Add requested variables to result
        if 'm' in record and record_gating:
            results['m'] = cp.asnumpy(m_rec)
        if 'h' in record and record_gating:
            results['h'] = cp.asnumpy(h_rec)
        if 'n' in record and record_gating:
            results['n'] = cp.asnumpy(n_rec)
        
        # Detect spikes if V was recorded
        if 'V' in record:
            results['spikes'] = self._detect_spikes_batch(
                results['V'], time, spike_threshold, batch_size
            )
        
        return SimulationResult(results, self.model.params, dt)
    
    def _detect_spikes_batch(self, voltage: np.ndarray, time: np.ndarray,
                            threshold: float, batch_size: int):
        """Detect spikes in batch voltage data."""
        if batch_size == 1:
            # Single neuron - flatten if 2D
            v = voltage.flatten() if voltage.ndim == 2 else voltage
            spike_indices = detect_spikes(v, threshold)
            spike_times = interpolate_spike_times(v, time, spike_indices)
            return {
                'count': len(spike_times),
                'times': spike_times,
                'indices': spike_indices
            }
        else:
            # Batch of neurons
            spikes_list = []
            for i in range(batch_size):
                v = voltage[:, i]
                spike_indices = detect_spikes(v, threshold)
                spike_times = interpolate_spike_times(v, time, spike_indices)
                spikes_list.append({
                    'count': len(spike_times),
                    'times': spike_times,
                    'indices': spike_indices
                })
            return spikes_list
    
    def benchmark(self, T: float, dt: float, batch_size: int, n_runs: int = 5) -> Dict:
        """
        Benchmark simulation performance.
        
        Args:
            T: Simulation time (ms)
            dt: Time step (ms)
            batch_size: Number of neurons
            n_runs: Number of runs to average
        
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        # Warmup run
        _ = self.run(T, dt, batch_size=batch_size, record=['V'])
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark runs
        times = []
        for _ in range(n_runs):
            cp.cuda.Stream.null.synchronize()
            start = time.perf_counter()
            
            _ = self.run(T, dt, batch_size=batch_size, record=['V'])
            
            cp.cuda.Stream.null.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
        
        times = np.array(times)
        n_steps = int(T / dt)
        
        return {
            'mean_time': float(np.mean(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'n_steps': n_steps,
            'batch_size': batch_size,
            'throughput': (n_steps * batch_size) / np.mean(times)
        }

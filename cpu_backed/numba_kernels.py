"""
Numba-accelerated kernels for HH simulation.

This module provides Numba-jitted implementations optimized for
small batch sizes or single neuron simulations where Python overhead matters.
"""

import numpy as np
from numba import njit, prange
from typing import Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Numba-jitted gating functions
@njit
def alpha_m(V):
    """Sodium activation rate (m gate)."""
    x = V + 40.0
    if abs(x) < 1e-4:
        return 1.0
    return 0.1 * x / (1.0 - np.exp(-x / 10.0))


@njit
def beta_m(V):
    """Sodium activation rate (m gate)."""
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


@njit
def alpha_h(V):
    """Sodium inactivation rate (h gate)."""
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


@njit
def beta_h(V):
    """Sodium inactivation rate (h gate)."""
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


@njit
def alpha_n(V):
    """Potassium activation rate (n gate)."""
    x = V + 55.0
    if abs(x) < 1e-4:
        return 0.1
    return 0.01 * x / (1.0 - np.exp(-x / 10.0))


@njit
def beta_n(V):
    """Potassium activation rate (n gate)."""
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


@njit
def compute_derivatives_single(V, m, h, n, I_ext, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L):
    """
    Compute derivatives for a single neuron.
    
    Returns: (dV, dm, dh, dn)
    """
    # Gating rates
    a_m = alpha_m(V)
    b_m = beta_m(V)
    a_h = alpha_h(V)
    b_h = beta_h(V)
    a_n = alpha_n(V)
    b_n = beta_n(V)
    
    # Gating derivatives
    dm = a_m * (1.0 - m) - b_m * m
    dh = a_h * (1.0 - h) - b_h * h
    dn = a_n * (1.0 - n) - b_n * n
    
    # Ionic currents
    I_Na = g_Na * (m ** 3) * h * (V - E_Na)
    I_K = g_K * (n ** 4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    # Voltage derivative
    dV = (I_ext - I_Na - I_K - I_L) / C_m
    
    return dV, dm, dh, dn


@njit
def rk4_step_single(V, m, h, n, I_ext, dt, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L):
    """
    Single RK4 step for one neuron.
    
    Returns: (V_new, m_new, h_new, n_new)
    """
    # k1
    dV1, dm1, dh1, dn1 = compute_derivatives_single(
        V, m, h, n, I_ext, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
    )
    
    # k2
    V2 = V + 0.5 * dt * dV1
    m2 = m + 0.5 * dt * dm1
    h2 = h + 0.5 * dt * dh1
    n2 = n + 0.5 * dt * dn1
    dV2, dm2, dh2, dn2 = compute_derivatives_single(
        V2, m2, h2, n2, I_ext, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
    )
    
    # k3
    V3 = V + 0.5 * dt * dV2
    m3 = m + 0.5 * dt * dm2
    h3 = h + 0.5 * dt * dh2
    n3 = n + 0.5 * dt * dn2
    dV3, dm3, dh3, dn3 = compute_derivatives_single(
        V3, m3, h3, n3, I_ext, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
    )
    
    # k4
    V4 = V + dt * dV3
    m4 = m + dt * dm3
    h4 = h + dt * dh3
    n4 = n + dt * dn3
    dV4, dm4, dh4, dn4 = compute_derivatives_single(
        V4, m4, h4, n4, I_ext, C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
    )
    
    # Combine
    V_new = V + (dt / 6.0) * (dV1 + 2*dV2 + 2*dV3 + dV4)
    m_new = m + (dt / 6.0) * (dm1 + 2*dm2 + 2*dm3 + dm4)
    h_new = h + (dt / 6.0) * (dh1 + 2*dh2 + 2*dh3 + dh4)
    n_new = n + (dt / 6.0) * (dn1 + 2*dn2 + 2*dn3 + dn4)
    
    return V_new, m_new, h_new, n_new


@njit
def simulate_single_neuron(V0, m0, h0, n0, I_ext_array, dt, n_steps,
                          C_m, g_Na, g_K, g_L, E_Na, E_K, E_L):
    """
    Simulate single neuron with Numba acceleration.
    
    Args:
        V0, m0, h0, n0: Initial state
        I_ext_array: External current at each time step
        dt: Time step
        n_steps: Number of steps
        C_m, g_Na, g_K, g_L, E_Na, E_K, E_L: Parameters
    
    Returns:
        Tuple of arrays (V, m, h, n) for all time steps
    """
    # Allocate output arrays
    V_out = np.zeros(n_steps + 1)
    m_out = np.zeros(n_steps + 1)
    h_out = np.zeros(n_steps + 1)
    n_out = np.zeros(n_steps + 1)
    
    # Initial state
    V_out[0] = V0
    m_out[0] = m0
    h_out[0] = h0
    n_out[0] = n0
    
    V, m, h, n = V0, m0, h0, n0
    
    # Time stepping
    for i in range(n_steps):
        I_ext = I_ext_array[i]
        V, m, h, n = rk4_step_single(V, m, h, n, I_ext, dt, 
                                     C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)
        V_out[i + 1] = V
        m_out[i + 1] = m
        h_out[i + 1] = h
        n_out[i + 1] = n
    
    return V_out, m_out, h_out, n_out


@njit(parallel=True)
def simulate_batch_parallel(state0, I_ext_array, dt, n_steps,
                           C_m, g_Na, g_K, g_L, E_Na, E_K, E_L):
    """
    Simulate batch of neurons with parallel Numba acceleration.
    
    Args:
        state0: Initial state array (batch_size, 4)
        I_ext_array: External current (n_steps, batch_size)
        dt: Time step
        n_steps: Number of steps
        C_m, g_Na, g_K, g_L, E_Na, E_K, E_L: Parameters
    
    Returns:
        State array (n_steps+1, batch_size, 4)
    """
    batch_size = state0.shape[0]
    
    # Allocate output
    state_out = np.zeros((n_steps + 1, batch_size, 4))
    state_out[0, :, :] = state0
    
    # Parallel loop over batch
    for b in prange(batch_size):
        V = state0[b, 0]
        m = state0[b, 1]
        h = state0[b, 2]
        n = state0[b, 3]
        
        for i in range(n_steps):
            I_ext = I_ext_array[i, b]
            V, m, h, n = rk4_step_single(V, m, h, n, I_ext, dt,
                                        C_m, g_Na, g_K, g_L, E_Na, E_K, E_L)
            state_out[i + 1, b, 0] = V
            state_out[i + 1, b, 1] = m
            state_out[i + 1, b, 2] = h
            state_out[i + 1, b, 3] = n
    
    return state_out


@njit
def rush_larsen_step_single(V, m, h, n, dt):
    """
    Rush-Larsen step for gating variables (single neuron).
    
    Returns: (m_new, h_new, n_new)
    """
    # Compute rates
    a_m = alpha_m(V)
    b_m = beta_m(V)
    a_h = alpha_h(V)
    b_h = beta_h(V)
    a_n = alpha_n(V)
    b_n = beta_n(V)
    
    # Steady states and time constants
    m_inf = a_m / (a_m + b_m)
    tau_m = 1.0 / (a_m + b_m)
    
    h_inf = a_h / (a_h + b_h)
    tau_h = 1.0 / (a_h + b_h)
    
    n_inf = a_n / (a_n + b_n)
    tau_n = 1.0 / (a_n + b_n)
    
    # Exponential integration
    m_new = m_inf + (m - m_inf) * np.exp(-dt / tau_m)
    h_new = h_inf + (h - h_inf) * np.exp(-dt / tau_h)
    n_new = n_inf + (n - n_inf) * np.exp(-dt / tau_n)
    
    return m_new, h_new, n_new


# Python wrapper classes for Numba kernels

class NumbaSimulator:
    """
    Simulator using Numba-accelerated kernels.
    
    Best for single neurons or small batches where Python overhead
    would otherwise dominate.
    """
    
    def __init__(self, C_m=1.0, g_Na=120.0, g_K=36.0, g_L=0.3,
                 E_Na=50.0, E_K=-77.0, E_L=-54.387):
        """Initialize with HH parameters."""
        self.C_m = C_m
        self.g_Na = g_Na
        self.g_K = g_K
        self.g_L = g_L
        self.E_Na = E_Na
        self.E_K = E_K
        self.E_L = E_L
    
    def run_single(self, V0, m0, h0, n0, I_ext, dt, T):
        """
        Run simulation for single neuron.
        
        Args:
            V0, m0, h0, n0: Initial state
            I_ext: External current array
            dt: Time step (ms)
            T: Total time (ms)
        
        Returns:
            Dictionary with 'V', 'm', 'h', 'n', 'time'
        """
        n_steps = int(np.ceil(T / dt))
        time = np.linspace(0, T, n_steps + 1)
        
        # Ensure I_ext has correct length
        if len(I_ext) < n_steps:
            I_ext = np.pad(I_ext, (0, n_steps - len(I_ext)), mode='constant')
        
        V, m, h, n = simulate_single_neuron(
            V0, m0, h0, n0, I_ext, dt, n_steps,
            self.C_m, self.g_Na, self.g_K, self.g_L,
            self.E_Na, self.E_K, self.E_L
        )
        
        return {
            'time': time,
            'V': V,
            'm': m,
            'h': h,
            'n': n
        }
    
    def run_batch(self, state0, I_ext, dt, T):
        """
        Run simulation for batch of neurons with parallel Numba.
        
        Args:
            state0: Initial state (batch_size, 4)
            I_ext: External current (n_steps, batch_size)
            dt: Time step (ms)
            T: Total time (ms)
        
        Returns:
            Dictionary with results
        """
        n_steps = int(np.ceil(T / dt))
        time = np.linspace(0, T, n_steps + 1)
        
        # Ensure I_ext has correct shape
        batch_size = state0.shape[0]
        if I_ext.shape[0] < n_steps:
            pad_width = ((0, n_steps - I_ext.shape[0]), (0, 0))
            I_ext = np.pad(I_ext, pad_width, mode='constant')
        
        state_out = simulate_batch_parallel(
            state0, I_ext, dt, n_steps,
            self.C_m, self.g_Na, self.g_K, self.g_L,
            self.E_Na, self.E_K, self.E_L
        )
        
        return {
            'time': time,
            'V': state_out[:, :, 0].T,  # (batch_size, n_steps+1)
            'm': state_out[:, :, 1].T,
            'h': state_out[:, :, 2].T,
            'n': state_out[:, :, 3].T
        }

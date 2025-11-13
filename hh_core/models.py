"""
Hodgkin-Huxley model equations, gating kinetics, and parameter definitions.
"""

import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class HHParameters:
    """
    Parameters for the Hodgkin-Huxley neuron model.
    
    Default values are based on the original 1952 paper (scaled for mV, ms, uA/cm^2).
    """
    # Membrane capacitance (uF/cm^2)
    C_m: float = 1.0
    
    # Maximal conductances (mS/cm^2)
    g_Na: float = 120.0
    g_K: float = 36.0
    g_L: float = 0.3
    
    # Reversal potentials (mV)
    E_Na: float = 50.0
    E_K: float = -77.0
    E_L: float = -54.387
    
    # Temperature-related (for alpha/beta functions)
    # Original HH at 6.3°C; scale factors if needed
    temp_celsius: float = 6.3
    
    def to_dict(self) -> Dict[str, float]:
        """Convert parameters to dictionary."""
        return {
            'C_m': self.C_m,
            'g_Na': self.g_Na,
            'g_K': self.g_K,
            'g_L': self.g_L,
            'E_Na': self.E_Na,
            'E_K': self.E_K,
            'E_L': self.E_L,
            'temp_celsius': self.temp_celsius
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> 'HHParameters':
        """Create parameters from dictionary."""
        return cls(**d)


@dataclass
class HHState:
    """
    State variables for the Hodgkin-Huxley model.
    
    Shape conventions:
    - Single neuron: (4,) array [V, m, h, n]
    - Batch of neurons: (batch_size, 4) array
    """
    data: np.ndarray  # shape: (batch_size, 4) or (4,)
    
    @property
    def V(self) -> np.ndarray:
        """Membrane potential (mV)."""
        return self.data[..., 0]
    
    @property
    def m(self) -> np.ndarray:
        """Sodium activation gating variable."""
        return self.data[..., 1]
    
    @property
    def h(self) -> np.ndarray:
        """Sodium inactivation gating variable."""
        return self.data[..., 2]
    
    @property
    def n(self) -> np.ndarray:
        """Potassium activation gating variable."""
        return self.data[..., 3]
    
    @V.setter
    def V(self, value):
        self.data[..., 0] = value
    
    @m.setter
    def m(self, value):
        self.data[..., 1] = value
    
    @h.setter
    def h(self, value):
        self.data[..., 2] = value
    
    @n.setter
    def n(self, value):
        self.data[..., 3] = value
    
    @staticmethod
    def resting_state(batch_size: int = 1, dtype=np.float64) -> 'HHState':
        """
        Create initial state at rest.
        
        Uses steady-state values at V = -65 mV (typical resting potential).
        """
        V_rest = -65.0
        
        # Compute steady-state gating variables at rest
        alpha_m = alpha_m_func(V_rest)
        beta_m = beta_m_func(V_rest)
        m_inf = alpha_m / (alpha_m + beta_m)
        
        alpha_h = alpha_h_func(V_rest)
        beta_h = beta_h_func(V_rest)
        h_inf = alpha_h / (alpha_h + beta_h)
        
        alpha_n = alpha_n_func(V_rest)
        beta_n = beta_n_func(V_rest)
        n_inf = alpha_n / (alpha_n + beta_n)
        
        # Create state array
        if batch_size == 1:
            data = np.array([V_rest, m_inf, h_inf, n_inf], dtype=dtype)
        else:
            data = np.tile([V_rest, m_inf, h_inf, n_inf], (batch_size, 1)).astype(dtype)
        
        return HHState(data)


# Gating variable rate functions (alpha and beta)
# Following Hodgkin & Huxley 1952 formulation

def alpha_m_func(V: np.ndarray) -> np.ndarray:
    """
    Sodium activation rate (m gate).
    
    alpha_m = 0.1 * (V + 40) / (1 - exp(-(V + 40) / 10))
    """
    x = V + 40.0
    # Handle singularity at V = -40 using L'Hôpital's rule
    result = np.where(
        np.abs(x) < 1e-4,
        1.0,  # limit as x->0
        0.1 * x / (1.0 - np.exp(-x / 10.0))
    )
    return result


def beta_m_func(V: np.ndarray) -> np.ndarray:
    """
    Sodium activation rate (m gate).
    
    beta_m = 4.0 * exp(-(V + 65) / 18)
    """
    return 4.0 * np.exp(-(V + 65.0) / 18.0)


def alpha_h_func(V: np.ndarray) -> np.ndarray:
    """
    Sodium inactivation rate (h gate).
    
    alpha_h = 0.07 * exp(-(V + 65) / 20)
    """
    return 0.07 * np.exp(-(V + 65.0) / 20.0)


def beta_h_func(V: np.ndarray) -> np.ndarray:
    """
    Sodium inactivation rate (h gate).
    
    beta_h = 1.0 / (1 + exp(-(V + 35) / 10))
    """
    return 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))


def alpha_n_func(V: np.ndarray) -> np.ndarray:
    """
    Potassium activation rate (n gate).
    
    alpha_n = 0.01 * (V + 55) / (1 - exp(-(V + 55) / 10))
    """
    x = V + 55.0
    # Handle singularity at V = -55 using L'Hôpital's rule
    result = np.where(
        np.abs(x) < 1e-4,
        0.1,  # limit as x->0
        0.01 * x / (1.0 - np.exp(-x / 10.0))
    )
    return result


def beta_n_func(V: np.ndarray) -> np.ndarray:
    """
    Potassium activation rate (n gate).
    
    beta_n = 0.125 * exp(-(V + 65) / 80)
    """
    return 0.125 * np.exp(-(V + 65.0) / 80.0)


def compute_currents(state: HHState, params: HHParameters) -> Dict[str, np.ndarray]:
    """
    Compute ionic currents for given state and parameters.
    
    Returns:
        Dictionary with keys: 'I_Na', 'I_K', 'I_L', 'I_ion' (total ionic current)
    """
    V = state.V
    m = state.m
    h = state.h
    n = state.n
    
    # Sodium current
    I_Na = params.g_Na * (m ** 3) * h * (V - params.E_Na)
    
    # Potassium current
    I_K = params.g_K * (n ** 4) * (V - params.E_K)
    
    # Leak current
    I_L = params.g_L * (V - params.E_L)
    
    # Total ionic current
    I_ion = I_Na + I_K + I_L
    
    return {
        'I_Na': I_Na,
        'I_K': I_K,
        'I_L': I_L,
        'I_ion': I_ion
    }


def derivatives(state: HHState, I_ext: np.ndarray, params: HHParameters) -> np.ndarray:
    """
    Compute time derivatives of state variables.
    
    Args:
        state: Current state (V, m, h, n)
        I_ext: External current injection (uA/cm^2), shape matches batch dimension
        params: Model parameters
    
    Returns:
        Array of derivatives with same shape as state.data
    """
    V = state.V
    m = state.m
    h = state.h
    n = state.n
    
    # Compute rate functions
    a_m = alpha_m_func(V)
    b_m = beta_m_func(V)
    a_h = alpha_h_func(V)
    b_h = beta_h_func(V)
    a_n = alpha_n_func(V)
    b_n = beta_n_func(V)
    
    # Gating variable derivatives
    dm = a_m * (1.0 - m) - b_m * m
    dh = a_h * (1.0 - h) - b_h * h
    dn = a_n * (1.0 - n) - b_n * n
    
    # Membrane potential derivative
    currents = compute_currents(state, params)
    dV = (I_ext - currents['I_ion']) / params.C_m
    
    # Stack derivatives
    if state.data.ndim == 1:
        return np.array([dV, dm, dh, dn])
    else:
        return np.stack([dV, dm, dh, dn], axis=1)


def rush_larsen_gating_step(state: HHState, dt: float) -> HHState:
    """
    Apply Rush-Larsen exponential integration for gating variables.
    
    For each gating variable x: dx/dt = alpha*(1-x) - beta*x
    Exact solution: x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau_x)
    where x_inf = alpha/(alpha+beta), tau_x = 1/(alpha+beta)
    
    This method is more stable than Euler for gating variables.
    
    Args:
        state: Current state
        dt: Time step
    
    Returns:
        New state with updated gating variables (V unchanged)
    """
    V = state.V
    
    # Compute rate functions
    a_m = alpha_m_func(V)
    b_m = beta_m_func(V)
    a_h = alpha_h_func(V)
    b_h = beta_h_func(V)
    a_n = alpha_n_func(V)
    b_n = beta_n_func(V)
    
    # Compute steady-state values and time constants
    m_inf = a_m / (a_m + b_m)
    tau_m = 1.0 / (a_m + b_m)
    
    h_inf = a_h / (a_h + b_h)
    tau_h = 1.0 / (a_h + b_h)
    
    n_inf = a_n / (a_n + b_n)
    tau_n = 1.0 / (a_n + b_n)
    
    # Apply exponential integration
    new_data = state.data.copy()
    new_state = HHState(new_data)
    
    new_state.m = m_inf + (state.m - m_inf) * np.exp(-dt / tau_m)
    new_state.h = h_inf + (state.h - h_inf) * np.exp(-dt / tau_h)
    new_state.n = n_inf + (state.n - n_inf) * np.exp(-dt / tau_n)
    
    return new_state

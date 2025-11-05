"""
Numerical integration methods for the Hodgkin-Huxley equations.
"""

import numpy as np
from typing import Callable, Optional, Union, Tuple
from .models import HHState, HHParameters, derivatives, rush_larsen_gating_step


class IntegratorBase:
    """Base class for ODE integrators."""
    
    def __init__(self, params: HHParameters):
        self.params = params
    
    def step(self, state: HHState, dt: float, I_ext: np.ndarray) -> HHState:
        """
        Advance state by one time step.
        
        Args:
            state: Current state
            dt: Time step
            I_ext: External current
        
        Returns:
            New state
        """
        raise NotImplementedError


class ForwardEuler(IntegratorBase):
    """
    Forward Euler integration (first-order).
    
    Simple but can be unstable for large dt.
    """
    
    def step(self, state: HHState, dt: float, I_ext: np.ndarray) -> HHState:
        """Forward Euler step: y(t+dt) = y(t) + dt * f(y(t))"""
        dy = derivatives(state, I_ext, self.params)
        new_data = state.data + dt * dy
        return HHState(new_data)


class RK4(IntegratorBase):
    """
    Fourth-order Runge-Kutta integration (RK4).
    
    Provides good accuracy with reasonable computational cost.
    This is the recommended integrator for HH simulations.
    """
    
    def step(self, state: HHState, dt: float, I_ext: np.ndarray) -> HHState:
        """
        RK4 step:
        k1 = f(y)
        k2 = f(y + dt/2 * k1)
        k3 = f(y + dt/2 * k2)
        k4 = f(y + dt * k3)
        y(t+dt) = y(t) + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        """
        # k1
        k1 = derivatives(state, I_ext, self.params)
        
        # k2
        state_k2 = HHState(state.data + 0.5 * dt * k1)
        k2 = derivatives(state_k2, I_ext, self.params)
        
        # k3
        state_k3 = HHState(state.data + 0.5 * dt * k2)
        k3 = derivatives(state_k3, I_ext, self.params)
        
        # k4
        state_k4 = HHState(state.data + dt * k3)
        k4 = derivatives(state_k4, I_ext, self.params)
        
        # Combine
        new_data = state.data + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        return HHState(new_data)


class RK4RushLarsen(IntegratorBase):
    """
    RK4 for voltage, Rush-Larsen for gating variables.
    
    This hybrid approach uses RK4 for membrane potential and Rush-Larsen
    exponential integration for gating variables, which can be more stable
    and allow larger time steps.
    """
    
    def step(self, state: HHState, dt: float, I_ext: np.ndarray) -> HHState:
        """
        Operator splitting:
        1. Update V using RK4 with frozen gating variables
        2. Update gating variables using Rush-Larsen
        """
        # Step 1: Update V with RK4 (gating variables frozen)
        # We'll use a simplified approach - full RK4 on complete system
        # then replace gating variables with Rush-Larsen
        
        # Standard RK4 step
        k1 = derivatives(state, I_ext, self.params)
        state_k2 = HHState(state.data + 0.5 * dt * k1)
        k2 = derivatives(state_k2, I_ext, self.params)
        state_k3 = HHState(state.data + 0.5 * dt * k2)
        k3 = derivatives(state_k3, I_ext, self.params)
        state_k4 = HHState(state.data + dt * k3)
        k4 = derivatives(state_k4, I_ext, self.params)
        
        new_data = state.data + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        state_rk4 = HHState(new_data)
        
        # Step 2: Replace gating variables with Rush-Larsen update
        # Use the new V from RK4
        final_state = HHState(state_rk4.data.copy())
        V_new = state_rk4.V
        
        # Compute Rush-Larsen update at new voltage
        temp_state = HHState(state.data.copy())
        temp_state.V = V_new  # Set voltage to new value
        rl_state = rush_larsen_gating_step(temp_state, dt)
        
        # Keep V from RK4, gates from Rush-Larsen
        final_state.m = rl_state.m
        final_state.h = rl_state.h
        final_state.n = rl_state.n
        
        return final_state


# def simulate(
#     state0: HHState,
#     t_span: Tuple[float, float],
#     dt: float,
#     integrator: IntegratorBase,
#     stimulus: Union[np.ndarray, Callable[[float], np.ndarray]],
#     record_vars: Optional[list] = None
# ) -> dict:
#     """
#     Simulate the HH model over a time span.
    
#     Args:
#         state0: Initial state
#         t_span: Tuple of (t_start, t_end) in milliseconds
#         dt: Time step in milliseconds
#         integrator: Integrator instance to use
#         stimulus: Either a pre-computed array of currents (shape: n_steps x batch_size)
#                   or a callable that takes time and returns current
#         record_vars: List of variables to record ['V', 'm', 'h', 'n']
#                      If None, records all variables
    
#     Returns:
#         Dictionary containing:
#             - 'time': array of time points
#             - 'V': recorded voltage (if in record_vars)
#             - 'm', 'h', 'n': recorded gating variables (if in record_vars)
#             - 'state': final state
#     """
#     t_start, t_end = t_span
#     n_steps = int(np.ceil((t_end - t_start) / dt))
#     time = np.linspace(t_start, t_end, n_steps + 1)
    
#     # Determine what to record
#     if record_vars is None:
#         record_vars = ['V', 'm', 'h', 'n']
    
#     # Determine batch size
#     if state0.data.ndim == 1:
#         batch_size = 1
#         batch_shape = ()
#     else:
#         batch_size = state0.data.shape[0]
#         batch_shape = (batch_size,)
    
#     # Prepare stimulus
#     if callable(stimulus):
#         # Generate stimulus on-the-fly
#         def get_current(t):
#             return stimulus(t)
#     else:
#         # Pre-computed stimulus array
#         if stimulus.ndim == 1:
#             stimulus_array = stimulus
#             def get_current(t_idx):
#                 if t_idx >= len(stimulus_array):
#                     return np.zeros(batch_shape)
#                 curr = stimulus_array[t_idx]
#                 if batch_size > 1 and np.isscalar(curr):
#                     return np.full(batch_shape, curr)
#                 return curr
#         else:
#             stimulus_array = stimulus
#             def get_current(t_idx):
#                 if t_idx >= len(stimulus_array):
#                     return np.zeros(batch_shape)
#                 return stimulus_array[t_idx]
    
#     # Initialize recording arrays
#     results = {'time': time}
    
#     if 'V' in record_vars:
#         if batch_size == 1:
#             results['V'] = np.zeros(n_steps + 1)
#         else:
#             results['V'] = np.zeros((n_steps + 1, batch_size))
    
#     if 'm' in record_vars:
#         if batch_size == 1:
#             results['m'] = np.zeros(n_steps + 1)
#         else:
#             results['m'] = np.zeros((n_steps + 1, batch_size))
    
#     if 'h' in record_vars:
#         if batch_size == 1:
#             results['h'] = np.zeros(n_steps + 1)
#         else:
#             results['h'] = np.zeros((n_steps + 1, batch_size))
    
#     if 'n' in record_vars:
#         if batch_size == 1:
#             results['n'] = np.zeros(n_steps + 1)
#         else:
#             results['n'] = np.zeros((n_steps + 1, batch_size))
    
#     # Record initial state
#     if 'V' in record_vars:
#         results['V'][0] = state0.V
#     if 'm' in record_vars:
#         results['m'][0] = state0.m
#     if 'h' in record_vars:
#         results['h'][0] = state0.h
#     if 'n' in record_vars:
#         results['n'][0] = state0.n
    
#     # Time-stepping loop
#     state = state0
#     for i in range(n_steps):
#         # Get current for this time step
#         if callable(stimulus):
#             I_ext = get_current(time[i])
#         else:
#             I_ext = get_current(i)
        
#         # Ensure I_ext has correct shape
#         if batch_size > 1:
#             if np.isscalar(I_ext):
#                 I_ext = np.full(batch_shape, I_ext)
#             elif I_ext.shape != batch_shape:
#                 raise ValueError(f"I_ext shape {I_ext.shape} doesn't match batch shape {batch_shape}")
#         else:
#             if isinstance(I_ext, np.ndarray) and I_ext.size == 1:
#                 I_ext = float(I_ext)
        
#         # Take integration step
#         state = integrator.step(state, dt, I_ext)
        
#         # Record
#         if 'V' in record_vars:
#             results['V'][i + 1] = state.V
#         if 'm' in record_vars:
#             results['m'][i + 1] = state.m
#         if 'h' in record_vars:
#             results['h'][i + 1] = state.h
#         if 'n' in record_vars:
#             results['n'][i + 1] = state.n
    
#     results['state'] = state
#     return results

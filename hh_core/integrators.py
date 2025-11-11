"""
Numerical integration methods for the Hodgkin-Huxley equations.
"""

import numpy as np
from typing import Callable, Optional, Union, Tuple
from .models import HHState, HHParameters, derivatives, rush_larsen_gating_step

try:
    from scipy.integrate import RK45
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


class RK45Scipy(IntegratorBase):
    """
    Scipy's RK45 (Dormand-Prince) integrator.
    
    Uses scipy.integrate.RK45 adaptive step-size integrator.
    Note: This wrapper forces fixed time steps to match other integrators.
    """
    
    def __init__(self, params: HHParameters):
        super().__init__(params)
        if not SCIPY_AVAILABLE:
            raise ImportError(
                "scipy is not installed. Install it with: pip install scipy"
            )
    
    def step(self, state: HHState, dt: float, I_ext: np.ndarray) -> HHState:
        """
        Single RK45 step using scipy.
        
        We use scipy's RK45 integrator but force it to take exactly one step
        of size dt to match the behavior of other integrators.
        """
        # Define the derivative function for scipy
        def func(t, y):
            """Wrapper to match scipy's expected signature."""
            temp_state = HHState(y.reshape(state.data.shape))
            dy = derivatives(temp_state, I_ext, self.params)
            return dy.flatten()
        
        # Flatten state for scipy
        y0 = state.data.flatten()
        
        # Create RK45 integrator
        # We set max_step=dt to force exactly the step size we want
        solver = RK45(func, 0.0, y0, dt, max_step=dt, first_step=dt)
        
        # Take one step
        solver.step()
        
        # Reshape back to original shape
        new_data = solver.y.reshape(state.data.shape)
        return HHState(new_data)

"""
HH Core - Core implementations of Hodgkin-Huxley equations and integrators.
"""

from .models import (
    HHParameters,
    HHState,
    alpha_m_func,
    alpha_h_func,
    alpha_n_func,
    beta_m_func,
    beta_h_func,
    beta_n_func,
    compute_currents,
    derivatives,
    rush_larsen_gating_step
)

from .integrators import (
    IntegratorBase,
    ForwardEuler,
    RK4,
    RK4RushLarsen
)

from .utils import (
    Stimulus,
    detect_spikes,
    interpolate_spike_times,
    compute_spike_statistics,
    compute_firing_rate,
    create_batched_stimulus
)

from .api import (
    HHModel,
    SimulationResult,
    BaseSimulator,
)

__all__ = [
    # Models
    'HHParameters',
    'HHState',
    'alpha_m_func',
    'alpha_h_func',
    'alpha_n_func',
    'beta_m_func',
    'beta_h_func',
    'beta_n_func',
    'compute_currents',
    'derivatives',
    'rush_larsen_gating_step',
    
    # Integrators
    'IntegratorBase',
    'ForwardEuler',
    'RK4',
    'RK4RushLarsen',
    
    # Utils
    'Stimulus',
    'detect_spikes',
    'interpolate_spike_times',
    'compute_spike_statistics',
    'compute_firing_rate',
    'create_batched_stimulus'
    
    # API
    'HHModel',
    'SimulationResult',
    'BaseSimulator',
]

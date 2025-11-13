"""
Pytest fixtures and configuration for HH tests.
"""

import pytest
import numpy as np
from cpu_backed import HHModel, CPUSimulator, Stimulus


@pytest.fixture
def default_model():
    """Fixture providing default HH model."""
    return HHModel()


@pytest.fixture
def cpu_simulator(default_model):
    """Fixture providing CPU simulator with RK4."""
    return CPUSimulator(model=default_model, integrator='rk4')


@pytest.fixture
def step_stimulus():
    """Fixture providing standard step stimulus."""
    return Stimulus.step(10.0, 10.0, 40.0, 100.0, 0.01)


@pytest.fixture
def constant_stimulus():
    """Fixture providing constant stimulus."""
    return Stimulus.constant(10.0, 100.0, 0.01)


@pytest.fixture(params=['rk4', 'rk4rl', 'euler'])
def all_integrators(request, default_model):
    """Fixture providing all available integrators."""
    return CPUSimulator(model=default_model, integrator=request.param)


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "physiological: mark test as checking physiological behavior"
    )
    config.addinivalue_line(
        "markers", "numerical: mark test as checking numerical properties"
    )
    config.addinivalue_line(
        "markers", "batch: mark test as checking batch functionality"
    )
    config.addinivalue_line(
        "markers", "scipy: mark test as requiring scipy"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU/CuPy"
    )
    config.addinivalue_line(
        "markers", "accuracy: mark test as checking numerical accuracy"
    )

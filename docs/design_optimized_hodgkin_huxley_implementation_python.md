# Design Document — Optimized Hodgkin–Huxley Implementation (Python)

## 1. Overview

This document describes the design and implementation plan for an **optimized Hodgkin–Huxley (HH) neuron model** in Python. The project will be developed in two major phases:

- **Phase A — CPU-optimized implementation**: a high-performance, vectorized, and well-tested pure‑Python (NumPy/Numba/Cython) implementation that runs efficiently on modern multicore CPUs.
- **Phase B — GPU acceleration**: a GPU-backed implementation for batch simulations using either CuPy / PyTorch / JAX or custom CUDA kernels to achieve large speedups for population simulations and parameter sweeps.

Goals:
- Correctness vs canonical HH behavior (match reference implementations like Brian2/NEURON at unit tolerance).
- High throughput for large numbers of neurons or long simulation times.
- Maintainable, modular API so research/production can reuse components.
- Clear benchmarking and reproducibility.

---

## 2. Requirements

### Functional
- Simulate single-compartment HH neurons (classic Na/K/Leak) with configurable parameters.
- Support variable stimuli (step, pulses, noisy currents, prerecorded time-series).
- Provide per-step outputs: membrane voltage, gating variables, ionic currents.
- Support batched simulations (N parallel neurons) with independent params OR shared parameters with per-neuron stimuli.
- Spike detection (threshold crossing) and basic spike statistics.

### Non-functional
- High performance (low CPU time per simulated ms for large batches).
- Numerical stability (configurable integrators and dt safeguards).
- Extensible to multi-compartment neurons and additional ion channels.
- Cross-platform portability (Linux, macOS, Windows) for CPU code; GPU code supports NVIDIA GPUs (CUDA), and potentially ROCm later.

---

## 3. High-level Architecture

```
hh_project/
├─ hh_core/                # core ODE models and integrators
│  ├─ models.py            # HH equations, gating kinetics, parameter structs
│  ├─ integrators.py       # RK4, Euler, exp-Euler, adaptive wrappers
│  └─ utils.py             # stimulus generators, spike detection
├─ cpu_backend/            # CPU-optimized routines
│  ├─ vectorized.py        # NumPy vectorized implementations
│  ├─ numba_kernels.py     # Numba-jitted hot paths
│  └─ cython/              # optional Cython versions for hotspots
├─ gpu_backend/            # GPU routines (CuPy/JAX/PyTorch + kernels)
│  ├─ cupy_impl.py
│  ├─ jax_impl.py
│  └─ cuda_kernels.cu      # custom CUDA kernels (optional)
├─ benchmarks/             # scripts to run benchmarks and collect results
├─ tests/                  # unit + integration + numerical validation
├─ notebooks/              # demos and profiling notebooks
└─ cli/                    # command-line runner and export utilities
```

Key components explained:
- **models.py** — contains parameterized HH model (C_m, g_Na, g_K, g_L, E_Na, E_K, E_L), gating `alpha`/`beta` functions, current functions, and a small `HHState` data structure.
- **integrators.py** — exposes same API for all integrators: `step(state, dt, I_ext)` and `simulate(state0, t_span, dt, integrator, stimulus)`.
- **vectorized.py** — vectorized (NumPy) ODE stepping allowing shape `(batch, variables)` to be advanced in a single call to maximize SIMD and memory locality.
- **numba_kernels.py** — numba `@njit` implementations for inner loop where Python overhead matters.
- **gpu_backend/** — array library-agnostic implementations that rely on a thin array-API layer (see Interoperability below).

---

## 4. Numerical Methods & ODE Handling

### 4.1 Problem characteristics
- The HH model is a system of **4 coupled, nonlinear, first-order ODEs** (V, m, h, n).
- Gating variables evolve on faster/slower timescales than V; stiffness is **mild** for typical parameters but can increase with modifications.
- The model is **not strongly stiff**, so explicit methods (RK4 / fixed-step) are usually adequate and common.

### 4.2 Integrator choices

**High priority (recommended for Phase A):**
- **Fixed-step RK4 (4th-order Runge–Kutta)** — excellent accuracy vs performance for fixed dt; easy to vectorize and stable with sufficiently small dt (e.g., dt=0.01 ms typical in literature).
- **Exponential Euler / Rush–Larsen for gating variables** — treat gating ODEs analytically over dt when they are linear in the gating variable (they are). This can improve stability and enable larger dt for gating variables.

**Optional / validation integrators:**
- **Adaptive RK45 (Dormand-Prince)** (via SciPy `solve_ivp`) for reference and accuracy checks (not for high-performance inner loop).
- **Implicit methods (BDF)** only if we observe stiffness in extended models (rare for classic HH).

### 4.3 Implementation details for RK4 (vectorized)
- Represent state as array shaped `(batch, 4)` i.e. columns `[V, m, h, n]`.
- Precompute alpha/beta using vectorized expressions to minimize Python overhead.
- For each RK stage, compute derivatives in pure NumPy or numba-jitted function; when on CPU prefer Numba for inner loops if batch size small and Python overhead dominates; prefer vectorized NumPy for large batch sizes to exploit BLAS/NumPy SIMD.
- Avoid temporaries where possible; reuse allocated arrays for k1..k4 to reduce allocations.
- Pseudocode (vectorized):

```python
# state: (B,4)
k1 = f(state, I_ext)
k2 = f(state + 0.5*dt*k1, I_ext)
k3 = f(state + 0.5*dt*k2, I_ext)
k4 = f(state + dt*k3, I_ext)
state_next = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
```

### 4.4 Exploiting Rush–Larsen for gating variables
Gating ODEs have the form `dx/dt = a(V)*(1-x) - b(V)*x = - (a+b) x + a` which is linear in `x` with time‑dependent coefficients (through V). Over a short time step dt with V treated as constant (or frozen at stage value), the exact update is:

```
x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt / tau_x)
where x_inf = a/(a+b), tau_x = 1/(a+b)
```

Use this to integrate gating variables inside each RK stage (or outside in operator-split manner) to increase stability and allow larger dt for gating variables.

### 4.5 Time stepping strategies
- Default: fixed dt with RK4 and Rush–Larsen for gates.
- Expose dt to user; implement validators for dt (warning if dt > recommended threshold).
- Support substepping for currents or stimuli if external inputs are high-frequency.

### 4.6 Spike/event detection
- Implement a vectorized threshold crossing detector: detect upward crossing of threshold (e.g. 0 mV or configurable) by checking `V[t] >= th & V[t-1] < th` for each neuron in batch.
- Optionally provide `interpolate_spike_time` using linear interpolation for higher timing precision.

---

## 5. CPU Backend Implementation Details

### 5.1 Two complementary implementations
- **NumPy-vectorized**: Best when batch size `B` is large (e.g. thousands). Use array operations, minimize Python loops, preallocate arrays.
- **Numba-jitted**: Best when `B` small or when we want to iterate in Python with small per-step overhead (e.g. single neuron long simulation). Use `@njit(parallel=True)` where appropriate; ensure types are explicit.

### 5.2 Memory layout and alignment
- Use `float64` by default for numerical accuracy; optionally provide `float32` mode for speed/memory.
- Ensure contiguous `C`-order arrays for fast vectorized ops and numba access.

### 5.3 Micro-optimizations
- Reuse preallocated temporary arrays (`k1..k4`, gating temps) to avoid repeated allocations.
- Fuse expressions where possible (e.g., compute alpha/beta arrays once per stage).
- Use `np.maximum`, `np.exp` vectorized forms instead of Python loops.

### 5.4 Parallelization
- Use NumPy + multiprocessing for embarrassingly parallel parameter sweeps (different seeds or parameter sets). Benchmark overhead vs Numba parallel loop.
- Use Numba's `prange` for inner loops if mixed Python+Numba approach chosen.

---

## 6. GPU Backend Implementation Details

### 6.1 High-level choices
- **Array library compatibility layer**: create a thin layer that exposes `xp` (`numpy` or `cupy` or `jax.numpy` or `torch`) so most model code can be backend-agnostic.
- **Recommended starting options**:
  - **CuPy** for a near-NumPy drop-in for CUDA GPU arrays (easy port of the NumPy pipeline).
  - **JAX** for auto-vectorization, XLA compiling, and automatic differentiation (powerful but API differs; good for gradients/parameter fitting).
  - **PyTorch** if you want PyTorch ecosystem (optimizers, autograd) and multi-GPU support.

Choose one to implement first — **CuPy** is simplest if the model is implemented with NumPy idioms and the goal is raw simulation speed.

### 6.2 Kernel strategy
- **Strategy A: Pure array operations (CuPy/JAX/PyTorch)**
  - Implement RK4+Rush–Larsen using the array-API; rely on the library to compile/launch fused operations.
  - Pros: fast to implement, leverages library optimizations.
  - Cons: may create temporaries causing extra memory traffic.

- **Strategy B: Custom CUDA kernels**
  - Implement a fused kernel that computes `state_next` from `state` and `I_ext` in one kernel call — reduces global memory reads/writes.
  - Use either `cupy.RawKernel` or pycuda / Numba CUDA to write kernels.
  - Pros: maximum performance, minimal memory traffic.
  - Cons: higher engineering cost; less portable; need careful debugging.

Recommendation: implement Strategy A first (CuPy/JAX). Profile and — if necessary — implement Strategy B fused kernels for hot loops.

### 6.3 Memory management on GPU
- Keep inputs (stimuli, params) resident on GPU for full simulations.
- Avoid round-trip host<->device transfers per timestep; only transfer summary data (spikes, occasional snapshots) or final traces if needed.
- Reuse GPU buffers for k1..k4 and temps.

### 6.4 Batch and packing strategy
- Layout state arrays as `(batch, variables)` in GPU memory — contiguous in the inner dimension to allow coalesced reads in CUDA kernels.
- Choose `batch` sizes to fill GPU SMs; allow variable-length batches using padding or segmented processing.

### 6.5 Multi-GPU / distributed
- For very large simulations, partition batches across GPUs with simple sharding; collect results at host.
- For parameter sweeps, run independent jobs per GPU for simplicity.

---

## 7. API Design

Public API should be small and consistent across CPU/GPU backends.

```python
from hh_optimized import HHModel, Simulator, Stimulus

model = HHModel(default_parameters)
sim = Simulator(model, backend='cpu', dtype='float64')
stim = Stimulus.step(amplitude=10.0, t0=10.0, t1=40.0)

result = sim.run(T=100.0, dt=0.01, state0=None, stimulus=stim, batch_shape=(1000,))
# result: object with arrays: result.time, result.V (batch, time), gating vars, spikes
```

Key features:
- `Simulator(..., backend='cpu'|'cupy'|'jax'|'torch')` chooses implementation.
- `run(..., record=['V','m','h','n'], spike_threshold=0.0)` to control memory footprint.
- `stepper` / `integrator` argument to swap integrators easily (RK4, forward Euler, Rush-Larsen-enabled RK4).

---

## 8. Verification & Validation

### 8.1 Unit tests
- Test `alpha`/`beta` consistent values with canonical formulas.
- Test Rush–Larsen closed-form against small-step Euler for gating ODEs.

### 8.2 Integration tests
- Reproduce classic HH traces (e.g. Hodgkin & Huxley 1952 figures) for a standard stimulus.
- Compare numeric traces to reference implementations (Brian2, `pyHH`) using L2 error thresholds.

### 8.3 Regression tests & CI
- Add CI that runs small simulations and checks spike counts / key statistics.
- Include speed regression tests on representative hardware (optional).

---

## 9. Benchmarking Plan

Metrics:
- **Throughput**: simulated ms per wall-clock second per neuron (or neurons/ms/s).
- **Latency**: time to simulate single neuron for long run (useful for real-time constraints).
- **Memory**: GPU/CPU memory used per neuron.

Benchmarks to run:
- Single neuron simulation at dt=0.01ms for T=1000ms.
- Batched simulations: B=1, 16, 128, 1024, 8192 to show scaling.
- Parameter sweep: vary dt and measure accuracy/perf tradeoffs.

Tools:
- `time.perf_counter()` in microbench harness.
- `line_profiler`, `pyinstrument`, `nvprof` / `nsys` for GPU profiling.

Reporting:
- Produce CSV/JSON benchmark outputs and a notebook that creates plots (throughput vs batch size, scaling curves).

---

## 10. Testing & Quality Assurance

- Continuous unit/integration tests via GitHub Actions.
- Linting (flake8/black/isort) and static typing (mypy).
- Documented examples and notebooks.

---

## 11. Extensibility

Planned extensions:
- Additional channels: calcium currents, M-type K+, H-currents.
- Multi-compartment cable equations (add axial current term) for spatially-extended neurons.
- Parameter fitting using autodiff-capable backend (JAX / PyTorch) to calibrate model to data.

Design choices made to enable extensibility:
- Backend-agnostic array layer (xp) so model code is portable.
- Modular integrator architecture.

---

## 12. Deliverables (Phase-based)

**Phase A – CPU**
- NumPy vectorized RK4 + Rush–Larsen implementation.
- Numba-jitted inner loops for small-batch speed.
- Unit tests, integration validation against reference traces.
- Benchmark suite and profiling notebook.

**Phase B – GPU**
- CuPy-based backend that reuses the same API.
- Benchmarks comparing CPU vs GPU across batch sizes.
- Optionally, fused CUDA kernel for maximal performance.

---

## 13. Risks and Mitigations

- **Risk**: Numerical instability at large dt.
  - *Mitigation*: Provide warnings, use Rush–Larsen, and recommend dt ranges. Provide adaptive-step reference runs.
- **Risk**: GPU memory exhaustion for very large batches.
  - *Mitigation*: Implement streaming, chunked processing, and efficient recording (only store spikes or sampled traces).
- **Risk**: Complexity of custom CUDA kernels.
  - *Mitigation*: Start with CuPy/JAX; only implement custom kernels when profiling shows bottleneck.

---

## 14. Open Questions / Decisions to Make

- Primary GPU backend: **CuPy** (fast porting) vs **JAX** (XLA, autodiff) vs **PyTorch** (ecosystem). Recommendation: start with CuPy; add JAX later if parameter estimation/autodiff is needed.
- Default dtype: `float64` or `float32`? Use `float64` for validation accuracy; provide `float32` mode for performance runs.
- Spike threshold and event precision: linear interpolation for spike time or accept sample resolution.

---

## 15. Example Pseudocode (Vectorized RK4 + Rush–Larsen)

```python
# state: (B,4) -> columns [V,m,h,n]
# I_ext: (B,) or (B,)

def derivatives(state, I_ext, params):
    V = state[:,0]
    m = state[:,1]
    h = state[:,2]
    n = state[:,3]
    # compute alpha/beta using vectorized ops
    a_m = alpha_m(V); b_m = beta_m(V)
    a_h = alpha_h(V); b_h = beta_h(V)
    a_n = alpha_n(V); b_n = beta_n(V)
    # gating derivatives
    dm = a_m*(1.0 - m) - b_m*m
    dh = a_h*(1.0 - h) - b_h*h
    dn = a_n*(1.0 - n) - b_n*n
    # currents
    INa = g_Na * (m**3) * h * (V - E_Na)
    IK  = g_K  * (n**4) * (V - E_K)
    IL  = g_L * (V - E_L)
    dV = (I_ext - INa - IK - IL) / C_m
    return stack([dV, dm, dh, dn], axis=1)

# RK4 step using derivatives
k1 = derivatives(state, I)
k2 = derivatives(state + 0.5*dt*k1, I)
k3 = derivatives(state + 0.5*dt*k2, I)
k4 = derivatives(state + dt*k3, I)
state_next = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)

# Apply Rush-Larsen for gating vars as optional refinement
# for gates x in [m,h,n]:
#   x_inf = a/(a+b); tau = 1/(a+b)
#   x_next = x_inf + (x - x_inf) * exp(-dt/tau)
```

---

## 16. Next steps
- Pick the CPU baseline implementation (NumPy + RK4 + Rush–Larsen) and implement minimal API + tests.
- Implement benchmarking harness and run baseline microbenchmarks.
- Add Numba optimizations based on profiler output.
- Port code to chosen GPU backend (CuPy) and run GPU benchmarks.




---

*End of design document.*


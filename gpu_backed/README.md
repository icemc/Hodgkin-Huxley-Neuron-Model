# GPU Implementation with Custom CUDA Kernels

This implementation uses custom CUDA kernels to parallelize Hodgkin-Huxley neuron simulations across GPUs using CuPy. The key insight is that while HH is sequential in **time**, it can be parallelized across **neurons** (spatial parallelism).

## Architecture

### The Parallelization Strategy

**CPU Approach:**
- Simulates neurons sequentially OR uses vectorized NumPy operations
- Limited to CPU cores (typically 4-16)

**GPU Approach (This Implementation):**
- Each CUDA thread simulates ONE neuron independently
- Thousands of neurons simulated in parallel
- Reuses exact same RK4 mathematics as CPU

### Custom CUDA Kernel

The implementation uses a custom CUDA kernel written in C++ that:

1. **Mirrors CPU RK4 logic exactly** - Same alpha/beta functions, same derivative calculations
2. **Thread parallelism** - Each thread (neuron) is independent
3. **SIMD execution** - All threads execute the same instructions on different data
4. **Grid/Block configuration** - Automatically calculates optimal thread arrangement

```
Grid: [num_blocks]
  ├─ Block 0: [256 threads] → Neurons 0-255
  ├─ Block 1: [256 threads] → Neurons 256-511
  └─ Block N: [remaining threads] → Remaining neurons

Each thread runs: RK4(neuron_id) for all time steps
```

## Key Features

### 1. Code Reuse
The CUDA kernel implements the **exact same** HH equations and RK4 integration as the CPU:
- Same gating variable rate functions (alpha_m, beta_m, etc.)
- Same RK4 stages (k1, k2, k3, k4)
- Same numerical precision (float32)

### 2. Efficient Memory Access
- **Structure-of-Arrays layout**: `[V1, V2, V3, ...]` not `[[V1,m1,h1,n1], [V2,m2,h2,n2], ...]`
- **Coalesced memory access**: Adjacent threads access adjacent memory
- **Double buffering**: Swap input/output buffers to avoid allocations

### 3. Minimal CPU-GPU Transfers
- Transfer stimulus once at start
- Keep all state on GPU during simulation
- Transfer results only at end

## Performance Results

Benchmark on NVIDIA GeForce RTX 3070 Ti (50ms simulation, dt=0.01ms):

| Batch Size | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| 1 neuron | 1.02s | 0.33s | **3.1x** |
| 10 neurons | 1.74s | 0.28s | **6.2x** |
| 100 neurons | 1.76s | 0.28s | **6.2x** |
| 1,000 neurons | 3.09s | 0.28s | **11.1x** |
| 10,000 neurons | 15.03s | 0.33s | **45.8x** |

### Key Observations

1. **GPU wins even for single neuron** (3x speedup) due to custom kernel efficiency
2. **Speedup scales with batch size** - GPU overhead is amortized
3. **GPU time stays nearly constant** (0.28-0.33s) regardless of batch size!
4. **CPU time scales linearly** with number of neurons
5. **Crossover point is immediate** - GPU beneficial for all batch sizes

## Usage

### Basic Single Neuron

```python
from gpu_backed import GPUSimulator

simulator = GPUSimulator()
result = simulator.run(T=100.0, dt=0.01, batch_size=1, 
                      stimulus=stimulus_array, record_all=True)

# Access results
time = result['time']
V = result['V'][:, 0]  # Voltage for neuron 0
m, h, n = result['m'][:, 0], result['h'][:, 0], result['n'][:, 0]
```

### Batch Simulation

```python
# Simulate 10,000 neurons with different stimuli
batch_size = 10000
stimulus = create_varied_stimuli(n_steps, batch_size)

result = simulator.run(T=50.0, dt=0.01, 
                      batch_size=batch_size,
                      stimulus=stimulus,
                      record_all=False)  # Save memory

# Result['V'] has shape (n_steps, 10000)
```

### Benchmarking

```python
# Compare CPU vs GPU
python benchmark_cpu_vs_gpu.py --n-runs 5 --batch-sizes 1 10 100 1000 10000
```

## Implementation Details

### CUDA Kernel Structure

```c++
__global__ void rk4_step_kernel(
    V_in, m_in, h_in, n_in,      // Input state
    V_out, m_out, h_out, n_out,  // Output state
    I_ext,                        // Stimulus
    dt,                          // Time step
    batch_size,                  // Number of neurons
    C_m, g_Na, g_K, g_L,        // Parameters
    E_Na, E_K, E_L
) {
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (neuron_id >= batch_size) return;
    
    // Load this neuron's state
    float V = V_in[neuron_id];
    float m = m_in[neuron_id];
    // ... etc
    
    // Compute RK4 step (same as CPU)
    // k1 = f(y)
    // k2 = f(y + dt/2 * k1)
    // k3 = f(y + dt/2 * k2)
    // k4 = f(y + dt * k3)
    // y_new = y + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    // Store result
    V_out[neuron_id] = V_new;
    // ... etc
}
```

### Python Integration

```python
# Compile kernel once at initialization
self.rk4_kernel = cp.RawKernel(CUDA_CODE, 'rk4_step_kernel')

# Compute grid size
blocks = (batch_size + 256 - 1) // 256
grid = (blocks,)
threads = (256,)

# Launch kernel for each time step
for i in range(n_steps):
    self.rk4_kernel(grid, threads, (
        V, m, h, n,              # Inputs
        V_out, m_out, h_out, n_out,  # Outputs
        I_ext[i], dt, batch_size,    # Parameters
        C_m, g_Na, g_K, g_L, E_Na, E_K, E_L
    ))
    
    # Swap buffers
    V, V_out = V_out, V
```

## Advantages

✅ **Exact same mathematics** as CPU implementation
✅ **Massive parallelism** - simulate 10,000+ neurons simultaneously
✅ **Scales efficiently** - GPU time nearly constant with batch size
✅ **Clean separation** - CUDA kernel is self-contained
✅ **Beneficial for all batch sizes** - even single neuron sees speedup

## Limitations

- **Float32 only** - Currently uses single precision (sufficient for HH)
- **Requires CUDA GPU** - NVIDIA GPU with CUDA support needed
- **CuPy dependency** - Must install cupy-cuda11x or cupy-cuda12x
- **Sequential in time** - Still advances through time steps one by one

## Future Improvements

Potential optimizations:
1. **Shared memory** - Use shared memory for parameters
2. **Occupancy optimization** - Tune block size for GPU architecture
3. **Multiple timesteps per kernel** - Reduce kernel launch overhead
4. **Float64 support** - Add double precision version
5. **Fused kernels** - Combine RK4 stages into single kernel call

## Comparison with CPU Implementation

| Aspect | CPU (NumPy) | GPU (CUDA Kernel) |
|--------|-------------|-------------------|
| **Parallelism** | Vectorization (SIMD) | Thread parallelism (SIMT) |
| **Max Neurons** | ~1000 practical | 10,000+ efficient |
| **Overhead** | Low | Moderate (kernel launch) |
| **Memory** | System RAM | GPU VRAM (8-24 GB) |
| **Precision** | float32/float64 | float32 |
| **Best For** | Small batches | Large batches |

## Conclusion

This GPU implementation demonstrates effective use of CUDA for parallelizing HH simulations:

1. **Reuses CPU logic** - Same RK4 algorithm, just parallelized
2. **SIMD parallelism** - Each thread = one neuron
3. **Excellent scaling** - 45x speedup for 10,000 neurons
4. **Production ready** - Clean API, efficient implementation

The key insight: **Parallelize across neurons (spatial), not time (temporal)**.

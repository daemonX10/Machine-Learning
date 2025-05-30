[video](https://youtu.be/9bBsvpg-Xlk?si=i-l0Y1FIZo96BHQQ)



# GPU Programming with Python: A Comprehensive Guide to Numba and CuPy

## Introduction to GPU Computing
GPU (Graphics Processing Unit) computing has revolutionized computational-intensive fields like machine learning, scientific computing, and data analysis. This guide covers the essentials of GPU programming using Python with a focus on two powerful libraries: Numba and CuPy.

## Why GPUs Matter for Machine Learning and Scientific Computing
- **Massive Parallelism**: Modern GPUs contain thousands of cores capable of simultaneous computation
- **Matrix Operations**: Neural networks rely heavily on matrix multiplications and other linear algebra operations that GPUs excel at
- **Performance Gain**: GPUs can provide 10-100x speedup for suitable workloads compared to CPUs
- **Energy Efficiency**: More computation per watt of power consumed
- **Cost Effectiveness**: Consumer GPUs offer excellent performance/cost ratio for many applications

## Understanding CPU vs GPU Architecture

### CPU Architecture
- **Design Philosophy**: Optimized for sequential processing and complex logic operations
- **Core Structure**:
  - Few powerful cores (typically 4-32 in modern systems)
  - High clock speeds (3-5 GHz)
  - Large cache memory (L1/L2/L3 caches)
  - Advanced branch prediction and out-of-order execution
- **Memory Access**: Low latency, optimized for random access patterns
- **Use Case Strength**: Complex algorithms with lots of branching logic and dependencies

### GPU Architecture
- **Design Philosophy**: Massive parallelism for data processing
- **Core Structure**:
  - Hundreds to thousands of simpler cores
  - Lower clock speeds (1-2 GHz) 
  - Smaller cache per core
  - Simplified control logic
- **Memory Access**: High bandwidth, optimized for sequential access patterns
- **Use Case Strength**: Data-parallel tasks where the same operation is applied to many data elements
- **Specialized Hardware**: Tensor cores for matrix operations in newer NVIDIA GPUs

### When to Use GPU vs CPU
- **Use GPU when**:
  - Processing large arrays or matrices
  - Performing the same operation across many data points
  - Working with tasks that can be parallelized
  - Running neural networks or complex simulations
- **Use CPU when**:
  - Executing sequential tasks with many dependencies
  - Running code with unpredictable branching
  - Working with small datasets
  - Performing operations that require low latency

## Python GPU Programming Options: Choosing the Right Tool

| Tool | Abstraction Level | Learning Curve | Use Case | Strengths | Limitations |
|------|------------------|----------------|----------|-----------|-------------|
| **TensorFlow/PyTorch** | Very High | Medium | Neural Networks, ML workflows | Pre-optimized operations, ecosystem support | Limited to ML operations |
| **CuPy** | High | Low (if familiar with NumPy) | Scientific computing, drop-in NumPy replacement | Easy transition from CPU code | Limited to NumPy-like operations |
| **Numba** | Medium | Medium | Custom algorithms, specific optimizations | Write CUDA kernels in Python | Not all Python features supported |
| **PyCUDA** | Low | High | Maximum control, custom GPU kernel development | Full access to CUDA features | Requires CUDA C knowledge |
| **CUDA C/C++** | Very Low | Very High | Complete hardware control, maximum performance | Best possible performance | Steep learning curve, not Python |
| **JAX** | High | Medium | Numerical computing, ML research | Automatic differentiation, JIT compilation | Newer ecosystem, evolving API |
| **Rapids** | High | Low | Data science workflows | Pandas-like API, end-to-end GPU pipelines | Focused on data science tasks |

### Choosing the Right Tool for Your Task
- **For beginners**: Start with CuPy if you're familiar with NumPy
- **For machine learning**: TensorFlow/PyTorch handle most GPU needs automatically
- **For data science**: Consider Rapids for GPU-accelerated dataframes and ML
- **For custom algorithms**: Numba offers a good balance of control and ease of use
- **For maximum performance**: PyCUDA or direct CUDA C/C++ may be necessary

## Getting Started with CuPy: NumPy for GPU

### What is CuPy?
CuPy is a NumPy-compatible array library accelerated with CUDA. It provides:
- Nearly identical API to NumPy for seamless transition
- GPU acceleration for numerical operations
- Integration with existing Python scientific stack

### Key Concepts in GPU Computing
- **Host vs Device**: 
  - **Host**: CPU and system RAM where Python code runs
  - **Device**: GPU and GPU memory where computations are accelerated
- **Memory Transfers**: Data must move between host and device (a major bottleneck)
- **Asynchronous Execution**: GPU operations can run asynchronously from CPU code

### Installation and Setup
```bash
# Install CuPy with pip (specify CUDA version)
pip install cupy-cuda11x  # For CUDA 11.x
# Or with conda
conda install -c conda-forge cupy
```

### Common CuPy Operations

```python
import numpy as np
import cupy as cp
import time

# Create array on CPU
array_cpu = np.random.randint(0, 255, size=(2000, 2000))

# Transfer to GPU (explicit memory transfer)
start = time.time()
array_gpu = cp.asarray(array_cpu)  
transfer_time = time.time() - start
print(f"Transfer time: {transfer_time:.4f} seconds")

# Create array directly on GPU (more efficient)
array_gpu = cp.random.randint(0, 255, size=(2000, 2000))

# Basic operations (just like NumPy)
result_gpu = cp.sin(array_gpu) * 2 + cp.sqrt(cp.abs(array_gpu))

# Transfer back to CPU when needed
array_back = cp.asnumpy(array_gpu)

# Use CuPy's implementation of SciPy functions
from cupyx.scipy import fft as fft_gpu
from scipy import fft as fft_cpu

# Compare performance
start = time.time()
result_cpu = fft_cpu.fftn(array_cpu)
cpu_time = time.time() - start

start = time.time()
result_gpu = fft_gpu.fftn(array_gpu)
gpu_time = time.time() - start

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time/gpu_time:.2f}x")

# Check results are similar (allowing for floating point differences)
result_from_gpu = cp.asnumpy(result_gpu)
max_diff = np.max(np.abs(result_cpu - result_from_gpu))
print(f"Maximum difference: {max_diff}")
```

### Advanced CuPy Features
- **Custom CUDA Kernels**: Use `cupy.ElementwiseKernel` for simple custom operations
- **Raw CUDA with CuPy**: Use `cupy.RawKernel` for more complex kernels
- **Multi-GPU Support**: Specify device with `with cp.cuda.Device(device_id):`
- **Memory Management**: Control memory with `cp.get_default_memory_pool()`

### CuPy Limitations and Considerations
- **Feature Coverage**: Not 100% of NumPy/SciPy is implemented
- **Memory Management**: GPUs have limited memory compared to system RAM
- **Overhead**: Small operations may be slower due to transfer overhead
- **Custom Algorithms**: Loops in Python remain slow - for these, consider Numba

### When to Use CuPy
- **Use when**: Working with large arrays, performing vectorized operations, or using NumPy-like workflows
- **Not ideal for**: Small datasets, complex branching logic, or operations requiring custom loops

## Numba CUDA Basics
- Just-in-time compiler that can target CUDA 
- Allows writing CUDA kernels in Python
- Check GPU information with `cuda.detect()`

### CUDA Programming Model
- **Grid**: The entire problem divided into blocks
- **Blocks**: Groups of threads that share memory
- **Threads**: Individual execution units that run the kernel
- **Kernel**: Function that executes on each thread

### Memory Hierarchy
- **Global memory**: Large, slow, accessible by all threads
- **Shared memory**: Fast, shared within a block
- **Local memory**: Small, private to each thread

### Simple Kernel Example (Add 1 to each matrix element)
```python
@cuda.jit
def add_one_kernel(A):
    # Get thread position
    x, y = cuda.grid(2)
    
    # Check if within bounds
    if x < A.shape[0] and y < A.shape[1]:
        A[x, y] += 1
```

### Matrix Multiplication Example
- Basic approach: Each thread calculates one element of output
- Optimized approach: Use shared memory to reduce global memory access
- Thread synchronization critical when using shared memory

### Executing Kernels
```python
from numba import cuda
import numpy as np

# Set up grid dimensions
threads_per_block = (16, 16)
blocks_per_grid = ((A.shape[0] + threads_per_block[0] - 1) // threads_per_block[0],
                  (A.shape[1] + threads_per_block[1] - 1) // threads_per_block[1])

# Execute kernel
my_kernel[blocks_per_grid, threads_per_block](arg1, arg2, ...)
```

## Best Practices
1. Minimize data transfers between host and device
2. Create arrays directly on GPU when possible
3. Use built-in CuPy functions when available
4. For custom algorithms, use Numba CUDA kernels
5. Always check array bounds in CUDA kernels
6. Optimize shared memory usage for performance
7. Use thread synchronization when modifying shared memory

## Requirements
- NVIDIA GPU with CUDA support
- NVIDIA drivers
- CUDA toolkit
- Alternative: OpenCL (for non-NVIDIA GPUs, but with fewer Python bindings)

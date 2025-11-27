

A modern, production-ready CUDA optimization library featuring high-performance GPU computing techniques with C++17/20, RAII resource management, and modular architecture.

## 🚀 Modern C++ CUDA Library

This library provides optimized CUDA implementations for fundamental GPU operations, featuring modern C++ design patterns, comprehensive benchmarking, and production-ready error handling.

**Creator**: [Anuj0x](https://github.com/Anuj0x) - Expert in Programming & Scripting Languages, Deep Learning & State-of-the-Art AI Models, Generative Models & Autoencoders, Advanced Attention Mechanisms & Model Optimization, Multimodal Fusion & Cross-Attention Architectures, Reinforcement Learning & Neural Architecture Search, AI Hardware Acceleration & MLOps, Computer Vision & Image Processing, Data Management & Vector Databases, Agentic LLMs & Prompt Engineering, Forecasting & Time Series Models, Optimization & Algorithmic Techniques, Blockchain & Decentralized Applications, DevOps, Cloud & Cybersecurity, Quantum AI & Circuit Design, Web Development Frameworks.

## ✨ Key Features

### **Modern C++17/20 Architecture**
- **RAII Memory Management**: Automatic CUDA resource cleanup with `CudaMemory<T>` and `HostMemory<T>`
- **Type Safety**: Templates, `enum class`, and strong typing throughout
- **Exception Safety**: Comprehensive error handling with detailed CUDA diagnostics
- **High-Precision Timing**: `<chrono>`-based performance measurement utilities

### **Production-Ready Design**
- **CMake Build System**: Cross-platform, modern build configuration
- **Modular Libraries**: Separate, reusable components for each operation type
- **Namespace Organization**: Clean `cuda_opt` namespace structure
- **Template Metaprogramming**: Generic, type-safe algorithm implementations

### **Performance Optimizations**
- **Elementwise Operations**: Vectorized memory access (`float`, `float2`, `float4`) with up to 93.8% bandwidth utilization
- **Reduction Operations**: 7 optimization levels from baseline to warp shuffle-based algorithms
- **Advanced Techniques**: Shared memory optimization, bank conflict avoidance, hierarchical parallelism

## 📊 Performance Benchmarks

### Elementwise Operations (V100 GPU)
| Vectorization | Bandwidth | Efficiency | Speedup |
|---------------|-----------|------------|---------|
| float         | 827 GB/s  | 91.9%      | 1.0x    |
| float2        | 838 GB/s  | 93.1%      | 1.1x    |
| float4        | 844 GB/s  | 93.8%      | 1.2x    |

### Reduction Operations (V100 GPU)
| Algorithm | Bandwidth | Speedup vs Baseline |
|-----------|-----------|-------------------|
| v1 (baseline) | ~600 GB/s | 1.0x            |
| v2 (no divergence) | ~750 GB/s | 1.25x          |
| v3 (no bank conflict) | ~800 GB/s | 1.33x        |
| v4 (add during load) | ~820 GB/s | 1.37x        |
| v5 (unroll last warp) | ~850 GB/s | 1.42x       |
| v6 (completely unroll) | ~870 GB/s | 1.45x      |
| v7 (shuffle) | **858 GB/s** | **1.43x**     |

## 🏗️ Project Structure

```
cuda-optima/
├── CMakeLists.txt              # Main CMake configuration
├── README.md                   # This documentation
├── include/cuda_opt/          # Header-only modern utilities
│   ├── cuda_utils.cuh         # RAII wrappers, timing, error handling
│   ├── elementwise.cuh        # Vectorized elementwise operations
│   └── reduce.cuh             # Multi-stage reduction algorithms
├── src/                       # Implementation files
│   ├── CMakeLists.txt
│   ├── elementwise/elementwise.cu
│   └── reduce/reduce.cu
└── examples/                  # Benchmark and demonstration programs
    ├── CMakeLists.txt
    ├── elementwise_example.cpp
    └── reduce_example.cpp
```

## 🛠️ Quick Start

### Prerequisites
- **CUDA Toolkit**: 11.0+
- **CMake**: 3.18+
- **C++ Compiler**: GCC 7+, Clang 5+, MSVC 2019+

### Build & Run

```bash
# Clone and build
git clone https://github.com/Anuj0x/cuda-optima.git
cd cuda-optima
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run benchmarks
./examples/elementwise_example
./examples/reduce_example
```

### Basic Usage

```cpp
#include "cuda_opt/elementwise.cuh"
#include "cuda_opt/reduce.cuh"

// RAII memory management - no manual cudaMalloc/cudaFree
cuda_opt::CudaMemory<float> d_a(N), d_b(N), d_c(N);
cuda_opt::HostMemory<float> h_a(N), h_b(N);

// Generate and transfer data
cuda_opt::generate_test_data(h_a.data(), h_b.data(), N);
d_a.copy_from_host(h_a.data());
d_b.copy_from_host(h_b.data());

// Execute with automatic error checking
cuda_opt::ElementwiseOperation<float> op(cuda_opt::ElementwiseConfig(4)); // float4
op.add(d_a.data(), d_b.data(), d_c.data(), N);
```

## 🔧 Configuration

### CUDA Architecture Targets
```cmake
# In CMakeLists.txt
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)  # Volta, Turing, Ampere, Ada
```

### Build Types
- `Release`: Optimized production builds
- `Debug`: Development with debugging symbols
- `RelWithDebInfo`: Release with debug information

## 📈 Advanced Features

- **Automatic Benchmarking**: Built-in performance measurement utilities
- **Result Verification**: CPU reference implementations for correctness testing
- **Multiple Optimization Levels**: Choose algorithms based on your performance needs
- **Stream Support**: Asynchronous execution with CUDA streams
- **Template Specialization**: Optimized implementations for different data types

## 🤝 Contributing

We welcome contributions! See our contributing guidelines for details on:
- Code style and architecture patterns
- Adding new optimization algorithms
- Performance benchmarking standards
- Documentation requiremenT
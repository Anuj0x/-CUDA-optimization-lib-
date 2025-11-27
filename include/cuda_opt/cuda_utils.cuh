#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <functional>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// CUDA kernel launch error checking
#define CUDA_CHECK_KERNEL() \
    do { \
        cudaError_t error = cudaGetLastError(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Kernel Launch Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
        error = cudaDeviceSynchronize(); \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA Kernel Execution Error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

namespace cuda_opt {

// RAII wrapper for CUDA device memory
template<typename T>
class CudaMemory {
public:
    explicit CudaMemory(size_t size) : size_(size), data_(nullptr) {
        CUDA_CHECK(cudaMalloc(&data_, size * sizeof(T)));
    }

    ~CudaMemory() {
        if (data_) {
            cudaFree(data_);
        }
    }

    // Disable copy
    CudaMemory(const CudaMemory&) = delete;
    CudaMemory& operator=(const CudaMemory&) = delete;

    // Enable move
    CudaMemory(CudaMemory&& other) noexcept : size_(other.size_), data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    CudaMemory& operator=(CudaMemory&& other) noexcept {
        if (this != &other) {
            if (data_) cudaFree(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    // Memory operations
    void copy_from_host(const T* host_data, size_t count = 0) {
        size_t copy_size = count == 0 ? size_ : count;
        CUDA_CHECK(cudaMemcpy(data_, host_data, copy_size * sizeof(T), cudaMemcpyHostToDevice));
    }

    void copy_to_host(T* host_data, size_t count = 0) const {
        size_t copy_size = count == 0 ? size_ : count;
        CUDA_CHECK(cudaMemcpy(host_data, data_, copy_size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }

private:
    size_t size_;
    T* data_;
};

// Host memory wrapper with automatic allocation/deallocation
template<typename T>
class HostMemory {
public:
    explicit HostMemory(size_t size) : size_(size), data_(new T[size]) {}
    ~HostMemory() { delete[] data_; }

    // Disable copy
    HostMemory(const HostMemory&) = delete;
    HostMemory& operator=(const HostMemory&) = delete;

    // Enable move
    HostMemory(HostMemory&& other) noexcept : size_(other.size_), data_(other.data_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }

    HostMemory& operator=(HostMemory&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    T* data() { return data_; }
    const T* data() const { return data_; }
    size_t size() const { return size_; }

    T& operator[](size_t index) { return data_[index]; }
    const T& operator[](size_t index) const { return data_[index]; }

private:
    size_t size_;
    T* data_;
};

// Timer class for performance measurements
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}

    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double elapsed_ms() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

    double elapsed_us() const {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// Benchmarking utility
template<typename Func>
double benchmark(Func&& func, int iterations = 100, bool warmup = true) {
    if (warmup) {
        for (int i = 0; i < 10; ++i) {
            func();
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    Timer timer;
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    return timer.elapsed_ms() / iterations;
}

// Vectorized memory access helpers
template<typename T>
__device__ __forceinline__ T load_float2(const T* ptr) {
    return reinterpret_cast<const float2*>(ptr)[0];
}

template<typename T>
__device__ __forceinline__ T load_float4(const T* ptr) {
    return reinterpret_cast<const float4*>(ptr)[0];
}

template<typename T>
__device__ __forceinline__ void store_float2(T* ptr, T value) {
    reinterpret_cast<float2*>(ptr)[0] = value;
}

template<typename T>
__device__ __forceinline__ void store_float4(T* ptr, T value) {
    reinterpret_cast<float4*>(ptr)[0] = value;
}

// Matrix offset calculation (row-major)
__host__ __device__ __forceinline__
size_t offset(size_t row, size_t col, size_t ld) {
    return row * ld + col;
}

// Verification utility
template<typename T>
bool verify_results(const T* result, const T* reference, size_t size, T epsilon = 1e-6) {
    for (size_t i = 0; i < size; ++i) {
        T diff = std::abs(result[i] - reference[i]);
        T rel_error = diff / std::max(std::abs(result[i]), std::abs(reference[i]));
        if (rel_error > epsilon) {
            std::cout << "Verification failed at index " << i
                      << ": result=" << result[i] << ", reference=" << reference[i]
                      << ", rel_error=" << rel_error << std::endl;
            return false;
        }
    }
    return true;
}

} // namespace cuda_opt

#pragma once

#include "cuda_utils.cuh"
#include <cstdint>

namespace cuda_opt {

// Elementwise operation types
enum class ElementwiseType {
    ADD,
    MULTIPLY,
    MAX,
    MIN
};

// Configuration for elementwise operations
struct ElementwiseConfig {
    static constexpr uint32_t THREADS_PER_BLOCK = 256;
    uint32_t vectorization_factor = 1; // 1, 2, or 4 for float2/float4

    ElementwiseConfig(uint32_t vec_factor = 1) : vectorization_factor(vec_factor) {}
};

// Elementwise addition kernel - scalar version
template<typename T>
__global__ void elementwise_add_scalar(const T* __restrict__ a,
                                     const T* __restrict__ b,
                                     T* __restrict__ c,
                                     size_t n) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// Elementwise addition kernel - vectorized float2 version
__global__ void elementwise_add_float2(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     size_t n) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < n) {
        const float2 a_val = load_float2<float2>(&a[idx]);
        const float2 b_val = load_float2<float2>(&b[idx]);
        const float2 c_val = make_float2(a_val.x + b_val.x, a_val.y + b_val.y);
        store_float2(&c[idx], c_val);
    }
}

// Elementwise addition kernel - vectorized float4 version
__global__ void elementwise_add_float4(const float* __restrict__ a,
                                     const float* __restrict__ b,
                                     float* __restrict__ c,
                                     size_t n) {
    const size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < n) {
        const float4 a_val = load_float4<float4>(&a[idx]);
        const float4 b_val = load_float4<float4>(&b[idx]);
        const float4 c_val = make_float4(a_val.x + b_val.x, a_val.y + b_val.y,
                                       a_val.z + b_val.z, a_val.w + b_val.w);
        store_float4(&c[idx], c_val);
    }
}

// Generic elementwise operation class
template<typename T>
class ElementwiseOperation {
public:
    explicit ElementwiseOperation(const ElementwiseConfig& config = ElementwiseConfig())
        : config_(config) {}

    // Execute elementwise addition
    void add(const T* d_a, const T* d_b, T* d_c, size_t n, cudaStream_t stream = 0) {
        const dim3 block(config_.THREADS_PER_BLOCK);
        const dim3 grid((n + config_.THREADS_PER_BLOCK - 1) / config_.THREADS_PER_BLOCK);

        if constexpr (std::is_same_v<T, float>) {
            switch (config_.vectorization_factor) {
                case 1:
                    elementwise_add_scalar<T><<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
                    break;
                case 2:
                    elementwise_add_float2<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
                    break;
                case 4:
                    elementwise_add_float4<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
                    break;
                default:
                    throw std::runtime_error("Unsupported vectorization factor");
            }
        } else {
            elementwise_add_scalar<T><<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
        }

        CUDA_CHECK_KERNEL();
    }

    // Benchmark different vectorization strategies for float
    template<typename Func>
    std::vector<double> benchmark_vectorization(Func&& setup_func, size_t n, int iterations = 100) {
        std::vector<double> results;

        // Test different vectorization factors
        for (uint32_t vec_factor : {1, 2, 4}) {
            config_.vectorization_factor = vec_factor;

            double time = benchmark([&]() {
                setup_func(*this);
            }, iterations);

            results.push_back(time);
        }

        return results;
    }

private:
    ElementwiseConfig config_;
};

// Utility function to generate test data
template<typename T>
void generate_test_data(T* a, T* b, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        a[i] = static_cast<T>(i % 456);
        b[i] = static_cast<T>(i % 13);
    }
}

// Utility function to compute reference result
template<typename T>
void compute_reference_add(const T* a, const T* b, T* c, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

} // namespace cuda_opt

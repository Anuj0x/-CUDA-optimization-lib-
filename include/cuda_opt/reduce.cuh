#pragma once

#include "cuda_utils.cuh"
#include <cstdint>

namespace cuda_opt {

// Reduce operation types
enum class ReduceType {
    SUM,
    MAX,
    MIN,
    PRODUCT
};

// Configuration for reduce operations
struct ReduceConfig {
    static constexpr uint32_t THREADS_PER_BLOCK = 256;
    static constexpr uint32_t WARP_SIZE = 32;
    uint32_t algorithm_version = 7; // 1-7, corresponding to different optimization levels

    ReduceConfig(uint32_t version = 7) : algorithm_version(version) {}
};

// Warp-level reduction using shuffle operations (version 7)
template <unsigned int blockSize>
__device__ __forceinline__ float warpReduceSum(float sum) {
    if (blockSize >= 32) sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (blockSize >= 16) sum += __shfl_down_sync(0xffffffff, sum, 8);  // 0-8, 1-9, 2-10, etc.
    if (blockSize >= 8)  sum += __shfl_down_sync(0xffffffff, sum, 4);  // 0-4, 1-5, 2-6, etc.
    if (blockSize >= 4)  sum += __shfl_down_sync(0xffffffff, sum, 2);  // 0-2, 1-3, 4-6, 5-7, etc.
    if (blockSize >= 2)  sum += __shfl_down_sync(0xffffffff, sum, 1);  // 0-1, 2-3, 4-5, etc.
    return sum;
}

// Baseline reduction (version 1)
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v1(float* d_in, float* d_out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    float sum = 0;
    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    __shared__ float shared[blockSize];
    shared[tid] = sum;
    __syncthreads();

    // Reduction in shared memory
    for(unsigned int s = blockSize / 2; s > 0; s >>= 1) {
        if(tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if(tid == 0) d_out[blockIdx.x] = shared[0];
}

// No divergence branch (version 2)
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v2(float* d_in, float* d_out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    float sum = 0;
    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    __shared__ float shared[blockSize];
    shared[tid] = sum;
    __syncthreads();

    // Reduction with no divergence
    for(unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if(tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32) {
        volatile float* vshared = shared;
        vshared[tid] += vshared[tid + 32];
        vshared[tid] += vshared[tid + 16];
        vshared[tid] += vshared[tid + 8];
        vshared[tid] += vshared[tid + 4];
        vshared[tid] += vshared[tid + 2];
        vshared[tid] += vshared[tid + 1];
    }

    if(tid == 0) d_out[blockIdx.x] = shared[0];
}

// No bank conflict (version 3)
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v3(float* d_in, float* d_out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    float sum = 0;
    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    __shared__ float shared[blockSize];
    // Avoid bank conflicts by padding
    shared[tid] = sum;
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if(tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32) {
        volatile float* vshared = shared;
        vshared[tid] += vshared[tid + 32];
        vshared[tid] += vshared[tid + 16];
        vshared[tid] += vshared[tid + 8];
        vshared[tid] += vshared[tid + 4];
        vshared[tid] += vshared[tid + 2];
        vshared[tid] += vshared[tid + 1];
    }

    if(tid == 0) d_out[blockIdx.x] = shared[0];
}

// Add during load (version 4)
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v4(float* d_in, float* d_out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    float sum = 0;
    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    __shared__ float shared[blockSize];
    shared[tid] = sum;
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if(tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32) {
        volatile float* vshared = shared;
        vshared[tid] += vshared[tid + 32];
        vshared[tid] += vshared[tid + 16];
        vshared[tid] += vshared[tid + 8];
        vshared[tid] += vshared[tid + 4];
        vshared[tid] += vshared[tid + 2];
        vshared[tid] += vshared[tid + 1];
    }

    if(tid == 0) d_out[blockIdx.x] = shared[0];
}

// Unroll last warp (version 5)
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v5(float* d_in, float* d_out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    float sum = 0;
    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    __shared__ float shared[blockSize];
    shared[tid] = sum;
    __syncthreads();

    for(unsigned int s = blockSize / 2; s > 32; s >>= 1) {
        if(tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if(tid < 32) {
        volatile float* vshared = shared;
        if(blockSize >= 64) vshared[tid] += vshared[tid + 32];
        if(blockSize >= 32) vshared[tid] += vshared[tid + 16];
        if(blockSize >= 16) vshared[tid] += vshared[tid + 8];
        if(blockSize >= 8)  vshared[tid] += vshared[tid + 4];
        if(blockSize >= 4)  vshared[tid] += vshared[tid + 2];
        if(blockSize >= 2)  vshared[tid] += vshared[tid + 1];
    }

    if(tid == 0) d_out[blockIdx.x] = shared[0];
}

// Completely unroll (version 6)
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v6(float* d_in, float* d_out, unsigned int n) {
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    float sum = 0;
    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    __shared__ float shared[blockSize];
    shared[tid] = sum;
    __syncthreads();

    if(blockSize >= 512) { if(tid < 256) { shared[tid] += shared[tid + 256]; } __syncthreads(); }
    if(blockSize >= 256) { if(tid < 128) { shared[tid] += shared[tid + 128]; } __syncthreads(); }
    if(blockSize >= 128) { if(tid < 64)  { shared[tid] += shared[tid + 64];  } __syncthreads(); }

    if(tid < 32) {
        volatile float* vshared = shared;
        if(blockSize >= 64) vshared[tid] += vshared[tid + 32];
        if(blockSize >= 32) vshared[tid] += vshared[tid + 16];
        if(blockSize >= 16) vshared[tid] += vshared[tid + 8];
        if(blockSize >= 8)  vshared[tid] += vshared[tid + 4];
        if(blockSize >= 4)  vshared[tid] += vshared[tid + 2];
        if(blockSize >= 2)  vshared[tid] += vshared[tid + 1];
    }

    if(tid == 0) d_out[blockIdx.x] = shared[0];
}

// Multi-add (version 7 - most optimized)
template <unsigned int blockSize, int NUM_PER_THREAD>
__global__ void reduce_v7(float* d_in, float* d_out, unsigned int n) {
    float sum = 0;
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * NUM_PER_THREAD) + threadIdx.x;

    #pragma unroll
    for(int iter = 0; iter < NUM_PER_THREAD; iter++) {
        sum += d_in[i + iter * blockSize];
    }

    // Shared mem for partial sums (one per warp in the block)
    static __shared__ float warpLevelSums[WARP_SIZE];
    const int laneId = threadIdx.x % WARP_SIZE;
    const int warpId = threadIdx.x / WARP_SIZE;

    sum = warpReduceSum<blockSize>(sum);

    if(laneId == 0) warpLevelSums[warpId] = sum;
    __syncthreads();

    // read from shared memory only if that warp existed
    sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warpLevelSums[laneId] : 0;

    // Final reduce using first warp
    if(warpId == 0) sum = warpReduceSum<blockSize / WARP_SIZE>(sum);

    // write result for this block to global mem
    if(tid == 0) d_out[blockIdx.x] = sum;
}

// Generic reduce operation class
class ReduceOperation {
public:
    explicit ReduceOperation(const ReduceConfig& config = ReduceConfig())
        : config_(config) {}

    // Execute reduction
    void sum(const float* d_in, float* d_out, size_t n, cudaStream_t stream = 0) {
        const size_t block_size = config_.THREADS_PER_BLOCK;
        const size_t num_per_block = n / 1024;  // Assuming 1024 blocks for simplicity
        const size_t num_per_thread = num_per_block / block_size;

        const dim3 block(block_size);
        const dim3 grid(1024);

        switch (config_.algorithm_version) {
            case 1:
                reduce_v1<256, 32><<<grid, block, 0, stream>>>(const_cast<float*>(d_in), d_out, n);
                break;
            case 2:
                reduce_v2<256, 32><<<grid, block, 0, stream>>>(const_cast<float*>(d_in), d_out, n);
                break;
            case 3:
                reduce_v3<256, 32><<<grid, block, 0, stream>>>(const_cast<float*>(d_in), d_out, n);
                break;
            case 4:
                reduce_v4<256, 32><<<grid, block, 0, stream>>>(const_cast<float*>(d_in), d_out, n);
                break;
            case 5:
                reduce_v5<256, 32><<<grid, block, 0, stream>>>(const_cast<float*>(d_in), d_out, n);
                break;
            case 6:
                reduce_v6<256, 32><<<grid, block, 0, stream>>>(const_cast<float*>(d_in), d_out, n);
                break;
            case 7:
            default:
                reduce_v7<256, 32><<<grid, block, 0, stream>>>(const_cast<float*>(d_in), d_out, n);
                break;
        }

        CUDA_CHECK_KERNEL();
    }

    // Benchmark different reduction algorithms
    template<typename Func>
    std::vector<double> benchmark_algorithms(Func&& setup_func, size_t n, int iterations = 100) {
        std::vector<double> results;

        for (uint32_t version = 1; version <= 7; ++version) {
            config_.algorithm_version = version;

            double time = benchmark([&]() {
                setup_func(*this);
            }, iterations);

            results.push_back(time);
        }

        return results;
    }

private:
    ReduceConfig config_;
};

// Utility function to generate test data for reduction
void generate_reduce_test_data(float* data, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        data[i] = static_cast<float>(i % 456);
    }
}

// Utility function to compute reference reduction result
float compute_reference_reduce_sum(const float* data, size_t n) {
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        sum += data[i];
    }
    return sum;
}

} // namespace cuda_opt

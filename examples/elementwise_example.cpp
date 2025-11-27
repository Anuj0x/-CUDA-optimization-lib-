#include "cuda_opt/elementwise.cuh"
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char** argv) {
    try {
        std::cout << "CUDA Elementwise Operations Example\n";
        std::cout << "===================================\n\n";

        // Configuration
        constexpr size_t N = 32 * 1024 * 1024; // Large dataset for meaningful benchmarking
        constexpr int ITERATIONS = 1000;

        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));

        // Get device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n";

        // Allocate memory using RAII wrappers
        cuda_opt::HostMemory<float> h_a(N);
        cuda_opt::HostMemory<float> h_b(N);
        cuda_opt::HostMemory<float> h_c(N);
        cuda_opt::HostMemory<float> h_ref(N);

        cuda_opt::CudaMemory<float> d_a(N);
        cuda_opt::CudaMemory<float> d_b(N);
        cuda_opt::CudaMemory<float> d_c(N);

        // Generate test data
        cuda_opt::generate_test_data(h_a.data(), h_b.data(), N);

        // Compute reference result on CPU
        cuda_opt::compute_reference_add(h_a.data(), h_b.data(), h_ref.data(), N);

        // Copy input data to device
        d_a.copy_from_host(h_a.data());
        d_b.copy_from_host(h_b.data());

        std::cout << "Benchmarking different vectorization strategies:\n";
        std::cout << std::setw(15) << "Vectorization" << std::setw(15) << "Time (ms)" << std::setw(15) << "Bandwidth (GB/s)" << std::endl;
        std::cout << std::string(45, '-') << std::endl;

        // Benchmark different vectorization strategies
        std::vector<std::string> vec_names = {"float", "float2", "float4"};
        std::vector<double> times;

        for (size_t i = 0; i < vec_names.size(); ++i) {
            uint32_t vec_factor = (i == 0) ? 1 : (i == 1) ? 2 : 4;
            cuda_opt::ElementwiseOperation<float> op(cuda_opt::ElementwiseConfig(vec_factor));

            // Benchmark
            double time = cuda_opt::benchmark([&]() {
                op.add(d_a.data(), d_b.data(), d_c.data(), N);
            }, ITERATIONS);

            times.push_back(time);

            // Calculate bandwidth (GB/s)
            // Each addition: 2 reads + 1 write = 12 bytes for float, 24 for float2, 48 for float4
            size_t bytes_per_element = 12 * vec_factor; // 2 inputs + 1 output per vectorized operation
            double bandwidth = (N * bytes_per_element / vec_factor) / (time / 1000.0) / (1024.0 * 1024.0 * 1024.0);

            std::cout << std::setw(15) << vec_names[i]
                      << std::setw(15) << std::fixed << std::setprecision(4) << time
                      << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth << std::endl;
        }

        // Verify results for the fastest version (float4)
        {
            cuda_opt::ElementwiseOperation<float> op(cuda_opt::ElementwiseConfig(4));
            op.add(d_a.data(), d_b.data(), d_c.data(), N);
            d_c.copy_to_host(h_c.data());

            if (cuda_opt::verify_results(h_c.data(), h_ref.data(), N)) {
                std::cout << "\n✓ Results verified successfully!\n";
            } else {
                std::cout << "\n✗ Results verification failed!\n";
                return EXIT_FAILURE;
            }
        }

        // Calculate speedup
        if (times.size() >= 2) {
            double speedup_float2 = times[0] / times[1];
            double speedup_float4 = times[0] / times[2];
            std::cout << "\nSpeedup compared to scalar (float):\n";
            std::cout << "float2: " << std::fixed << std::setprecision(2) << speedup_float2 << "x\n";
            std::cout << "float4: " << std::fixed << std::setprecision(2) << speedup_float4 << "x\n";
        }

        std::cout << "\nElementwise operations completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

#include "cuda_opt/reduce.cuh"
#include <iostream>
#include <iomanip>
#include <vector>

int main(int argc, char** argv) {
    try {
        std::cout << "CUDA Reduce Operations Example\n";
        std::cout << "==============================\n\n";

        // Configuration
        constexpr size_t N = 32 * 1024 * 1024; // Large dataset for meaningful benchmarking
        constexpr int ITERATIONS = 1000;
        constexpr size_t NUM_BLOCKS = 1024;

        // Initialize CUDA
        CUDA_CHECK(cudaSetDevice(0));

        // Get device properties
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        std::cout << "Using GPU: " << prop.name << std::endl;
        std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Total global memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n\n";

        // Allocate memory using RAII wrappers
        cuda_opt::HostMemory<float> h_input(N);
        cuda_opt::HostMemory<float> h_output(NUM_BLOCKS);
        cuda_opt::HostMemory<float> h_ref(NUM_BLOCKS);

        cuda_opt::CudaMemory<float> d_input(N);
        cuda_opt::CudaMemory<float> d_output(NUM_BLOCKS);

        // Generate test data
        cuda_opt::generate_reduce_test_data(h_input.data(), N);

        // Compute reference result (sum per block)
        for (size_t block = 0; block < NUM_BLOCKS; ++block) {
            float sum = 0.0f;
            const size_t block_start = block * (N / NUM_BLOCKS);
            const size_t block_end = block_start + (N / NUM_BLOCKS);
            for (size_t i = block_start; i < block_end; ++i) {
                sum += h_input[i];
            }
            h_ref[block] = sum;
        }

        // Copy input data to device
        d_input.copy_from_host(h_input.data());

        std::cout << "Benchmarking different reduction algorithms:\n";
        std::cout << std::setw(12) << "Algorithm" << std::setw(15) << "Time (ms)" << std::setw(15) << "Bandwidth (GB/s)" << std::endl;
        std::cout << std::string(42, '-') << std::endl;

        // Benchmark different reduction algorithms
        std::vector<std::string> algo_names = {"v1", "v2", "v3", "v4", "v5", "v6", "v7"};
        std::vector<double> times;

        for (size_t i = 0; i < algo_names.size(); ++i) {
            cuda_opt::ReduceOperation reduce_op(cuda_opt::ReduceConfig(i + 1));

            // Benchmark
            double time = cuda_opt::benchmark([&]() {
                reduce_op.sum(d_input.data(), d_output.data(), N);
            }, ITERATIONS);

            times.push_back(time);

            // Calculate bandwidth (GB/s)
            // Each element is read once and written once per block operation
            size_t bytes_processed = N * sizeof(float) + NUM_BLOCKS * sizeof(float);
            double bandwidth = bytes_processed / (time / 1000.0) / (1024.0 * 1024.0 * 1024.0);

            std::cout << std::setw(12) << algo_names[i]
                      << std::setw(15) << std::fixed << std::setprecision(4) << time
                      << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth << std::endl;
        }

        // Verify results for the fastest version (v7)
        {
            cuda_opt::ReduceOperation reduce_op(cuda_opt::ReduceConfig(7));
            reduce_op.sum(d_input.data(), d_output.data(), N);
            d_output.copy_to_host(h_output.data());

            if (cuda_opt::verify_results(h_output.data(), h_ref.data(), NUM_BLOCKS, 1e-3f)) {
                std::cout << "\n✓ Results verified successfully!\n";
            } else {
                std::cout << "\n✗ Results verification failed!\n";
                return EXIT_FAILURE;
            }
        }

        // Calculate speedup compared to baseline (v1)
        if (!times.empty()) {
            std::cout << "\nSpeedup compared to baseline (v1):\n";
            for (size_t i = 1; i < times.size(); ++i) {
                double speedup = times[0] / times[i];
                std::cout << algo_names[i] << ": " << std::fixed << std::setprecision(2) << speedup << "x\n";
            }

            // Calculate final reduction result
            float final_sum = 0.0f;
            for (size_t i = 0; i < NUM_BLOCKS; ++i) {
                final_sum += h_ref[i];
            }

            float expected_sum = cuda_opt::compute_reference_reduce_sum(h_input.data(), N);

            std::cout << "\nFinal reduction result:\n";
            std::cout << "Computed: " << std::fixed << std::setprecision(2) << final_sum << std::endl;
            std::cout << "Expected: " << std::fixed << std::setprecision(2) << expected_sum << std::endl;
            std::cout << "Difference: " << std::scientific << std::setprecision(2) << std::abs(final_sum - expected_sum) << std::endl;
        }

        std::cout << "\nReduce operations completed successfully!\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

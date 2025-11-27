#include "cuda_opt/elementwise.cuh"

// Explicit instantiations for commonly used types
template class cuda_opt::ElementwiseOperation<float>;
template class cuda_opt::ElementwiseOperation<double>;
template class cuda_opt::ElementwiseOperation<int>;

// Explicit instantiations for utility functions
template void cuda_opt::generate_test_data(float*, float*, size_t);
template void cuda_opt::generate_test_data(double*, double*, size_t);
template void cuda_opt::generate_test_data(int*, int*, size_t);

template void cuda_opt::compute_reference_add(const float*, const float*, float*, size_t);
template void cuda_opt::compute_reference_add(const double*, const double*, double*, size_t);
template void cuda_opt::compute_reference_add(const int*, const int*, int*, size_t);

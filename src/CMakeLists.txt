# CUDA utilities library
add_library(cuda_utils INTERFACE)
target_include_directories(cuda_utils INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(cuda_utils INTERFACE CUDA::cudart)

# Elementwise operations
add_library(elementwise
    elementwise/elementwise.cu
)
target_link_libraries(elementwise PRIVATE cuda_utils)
set_target_properties(elementwise PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Reduce operations
add_library(reduce
    reduce/reduce.cu
)
target_link_libraries(reduce PRIVATE cuda_utils)
set_target_properties(reduce PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# SGEMM operations
add_library(sgemm
    sgemm/sgemm.cu
)
target_link_libraries(sgemm PRIVATE cuda_utils CUDA::cublas)
set_target_properties(sgemm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# SGEMV operations
add_library(sgemv
    sgemv/sgemv.cu
)
target_link_libraries(sgemv PRIVATE cuda_utils)
set_target_properties(sgemv PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Install headers
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../include/ DESTINATION include)

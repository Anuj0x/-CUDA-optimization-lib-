# Elementwise examples
add_executable(elementwise_example elementwise_example.cpp)
target_link_libraries(elementwise_example PRIVATE elementwise cuda_utils)

# Reduce examples
add_executable(reduce_example reduce_example.cpp)
target_link_libraries(reduce_example PRIVATE reduce cuda_utils)

# SGEMM examples
add_executable(sgemm_example sgemm_example.cpp)
target_link_libraries(sgemm_example PRIVATE sgemm cuda_utils)

# SGEMV examples
add_executable(sgemv_example sgemv_example.cpp)
target_link_libraries(sgemv_example PRIVATE sgemv cuda_utils)

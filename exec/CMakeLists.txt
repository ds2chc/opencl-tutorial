add_executable(cl_info ${CMAKE_CURRENT_SOURCE_DIR}/cl_info.cc)
target_link_libraries(cl_info PRIVATE OpenCL::OpenCL)

add_executable(matmul_test ${CMAKE_CURRENT_SOURCE_DIR}/matmul_test.cc)
target_link_libraries(matmul_test PRIVATE OpenCL::OpenCL)
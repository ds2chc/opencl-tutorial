#include <iostream>
#include <chrono>

#include "ocl/ocl.h"
#include "utils.hpp"
#include "matrix.hpp"

using namespace ocl;


void matmul_naive(const Matrix& lhs, const Matrix& rhs, Matrix& result) {
  int M = lhs.Rows();
  int N = rhs.Cols();
  int K = lhs.Cols();

  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      float acc = 0;
      for (int k = 0; k < K; k++) {
        acc += lhs(m, k) * rhs(k, n);
      }
      result(m, n) = acc;
    }
  }
}

void BenchNaiveMatmul(const Matrix& lhs, 
                      const Matrix& rhs, 
                      const Matrix& ref,
                      size_t num_repeats=1) {
  double total_spent_time = 0.0;
  float total_err = 0.0f;

  for (int n = 0; n < num_repeats; n++) {
    Matrix result(lhs.Rows(), rhs.Cols());
    auto start = std::chrono::high_resolution_clock::now();
    matmul_naive(lhs, rhs, result);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double spent_time = diff.count();
    total_spent_time += spent_time;

    float err = 0.0f;
    for (int i = 0; i < result.NumElmts(); i++) {
      err += std::abs(result.RawPtr()[i] - ref.RawPtr()[i]);
    }
    total_err += err / static_cast<float>(lhs.Rows() * rhs.Cols());
  }

  std::cout << "<<< Naive kernel bench results >>>" << std::endl;
  std::cout << "spent time: " << total_spent_time << " seconds" << std::endl;
  std::cout << "err.: " << total_err << std::endl;
  std::cout << std::endl;
}

void BenchMatmulKernel_v1(const Matrix& lhs, 
                          const Matrix& rhs, 
                          const Matrix& ref,
                          Context& context,
                          CommandQueue& queue, 
                          Program& program,
                          size_t num_repeats=1) {
  int M = lhs.Rows(), N = rhs.Cols(), K = lhs.Cols();
  double total_spent_time = 0.0;
  float total_error = 0.0f;

  // Create kernel from program
  //auto kernel = clCreateKernel(program(), "SimpleKernel", &status);
  Kernel kernel(program, "matmul_v1");
  
  // Run kernel
  size_t num_threads = 4;
  size_t local_work_size[2] = { num_threads / 2, num_threads / 2};
  size_t global_work_offset[2] = { 0, 0 };
  size_t global_workers[2] = { (size_t)M, (size_t)N };

  for (int n = 0; n < num_repeats; n++) {
    std::vector<float> results(M * N, 0);
    
    // Create buffer
    Buffer<float> device_lhs(context, M * K);
    Buffer<float> device_rhs(context, K * N);
    Buffer<float> device_result(context, N * N);      
    device_lhs.CopyFromHost(queue, lhs.RawPtr(), lhs.NumElmts());
    device_rhs.CopyFromHost(queue, rhs.RawPtr(), rhs.NumElmts());
    
    auto start = std::chrono::high_resolution_clock::now();
    kernel.SetArguments(
      M, N, K,
      device_lhs(),
      device_rhs(),
      device_result()
    );
    kernel.Run(
      queue, 
      2, 
      global_work_offset,
      global_workers,
      local_work_size
    );
    queue.Finish();
    
    // Copy output buffer into host memory
    device_result.CopyFromDevice(queue, results.data(), M * N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double spent_time = diff.count();
    total_spent_time += spent_time;

    float err = 0.0f;
    for (int i = 0; i < M * N; i++) {
      err += std::abs(results[i] - ref.RawPtr()[i]);
    }
    total_error += err / static_cast<float>(M * N);
  }

  std::cout << "<<< Matrix multiplication kernel v1 bench results >>>" << std::endl;
  std::cout << "spent time: " << total_spent_time << " seconds" << std::endl;
  std::cout << "err.: " << total_error << std::endl;
  std::cout << std::endl;
}

void BenchMatmulKernel_v2(const Matrix& lhs, 
                          const Matrix& rhs, 
                          const Matrix& ref,
                          Context& context,
                          CommandQueue& queue, 
                          Program& program,
                          size_t num_repeats=1) {
  int M = lhs.Rows(), N = rhs.Cols(), K = lhs.Cols();
  double total_spent_time = 0.0;
  float total_error = 0.0f;

  // Create kernel from program
  //auto kernel = clCreateKernel(program(), "SimpleKernel", &status);
  Kernel kernel(program, "matmul_v2");
  
  // Run kernel
  size_t num_threads = 4;
  size_t local_work_size[2] = { num_threads / 2, num_threads / 2}; 
  size_t global_work_offset[2] = { 0, 0 };
  size_t global_workers[2] = { (size_t)M, (size_t)N };

  for (int n = 0; n < num_repeats; n++) {
    std::vector<float> results(M * N, 0);
    
    // Create buffer
    Buffer<float> device_lhs(context, M * K);
    Buffer<float> device_rhs(context, K * N);
    Buffer<float> device_result(context, N * N);      
    device_lhs.CopyFromHost(queue, lhs.RawPtr(), lhs.NumElmts());
    device_rhs.CopyFromHost(queue, rhs.RawPtr(), rhs.NumElmts());
    
    auto start = std::chrono::high_resolution_clock::now();
    kernel.SetArguments(
      M, N, K,
      device_lhs(),
      device_rhs(),
      device_result()
    );
    kernel.Run(
      queue, 
      2, 
      global_work_offset,
      global_workers,
      local_work_size
    );
    queue.Finish();
    
    // Copy output buffer into host memory
    device_result.CopyFromDevice(queue, results.data(), M * N);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    double spent_time = diff.count();
    total_spent_time += spent_time;

    float err = 0.0f;
    for (int i = 0; i < M * N; i++) {
      err += std::abs(results[i] - ref.RawPtr()[i]);
    }
    total_error += err / static_cast<float>(M * N);
  }

  std::cout << "<<< Matrix multiplication kernel v2 bench results >>>" << std::endl;
  std::cout << "spent time: " << total_spent_time << " seconds" << std::endl;
  std::cout << "err.: " << total_error << std::endl;
  std::cout << std::endl;

}

int main(int argc, char** argv) {
  int M = 1024, N = 1024, K = 8;
  size_t num_repeats = 1;

  // Initialize input matrix
  Matrix lhs(M, K);
  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      lhs(m, k) = (m + 1) * (k + 1) / static_cast<float>(M);
    }
  }

  Matrix rhs(K, N);
  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      rhs(k, n) = (k + 1) * (n + 1) / static_cast<float>(M);
    }
  }
  
  // Initialize reference
  Matrix reference(M, N);
  matmul_naive(lhs, rhs, reference);
  
  // Bench naive matrix multiplication kernel
  BenchNaiveMatmul(lhs, rhs, reference, num_repeats);

  // Bench matmul kernel v1
  std::string source_file_path = std::string(argv[1]);
  std::string source = utils::ReadKernelFileFromDisk(source_file_path);

  std::vector<Platform> all_platforms = GetAllPlatforms();
  size_t num_platforms = all_platforms.size();
  
  Device device(all_platforms[0], 0);

  cl_int status = 0;

  // create context 
  Context context(device);
  
  // Create command queue
  CommandQueue queue(context, device);
  
  // Compile & Build program
  Program program(context, source);
  program.Build(device, {""});
  
  BenchMatmulKernel_v1(
    lhs,
    rhs,
    reference,
    context,
    queue,
    program, 
    num_repeats
  );
  
  BenchMatmulKernel_v2(
    lhs,
    rhs,
    reference,
    context,
    queue,
    program,
    num_repeats
  );
  return 0;
}
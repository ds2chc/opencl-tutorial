#pragma once

//#include <numeric>

#include "ocl/common.h"


namespace ocl {

  class Program : public ObjectBase<cl_program> {
  public:
    explicit Program(const Context& context, const std::string& source) {
      const char* source_ptr = &source[0];
      const size_t length = source.length();

      cl_int status = CL_SUCCESS;
      object_ = clCreateProgramWithSource(
        context(),
        1,
        &source_ptr,
        &length,
        &status
      );
      if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program");
      }
    }

    /// @brief Create program from pre-built binary
    explicit Program(const Device& device, const Context& context, const std::string& binary) {
      const char *binary_ptr = &binary[0];
      const auto length = binary.length();
      auto status1 = CL_SUCCESS;
      auto status2 = CL_SUCCESS;
      const auto dev = device();
      object_ = clCreateProgramWithBinary(
        context(), 
        1, 
        &dev, 
        &length,
        reinterpret_cast<const unsigned char**>(&binary_ptr),
        &status1, &status2
      );
      if (status1 != CL_SUCCESS || status2 != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program");
      }
    }

    virtual ~Program() {}

    void Build(const Device& device, const std::vector<std::string>& options) {
      const cl_device_id d = device();

      cl_int err = clBuildProgram(
        object_, 1, &d, nullptr, nullptr, nullptr
      );
      if (err == CL_BUILD_PROGRAM_FAILURE) {
        // Determine the size of the log
        size_t log_size;
        clGetProgramBuildInfo(
          object_, 
          d, 
          CL_PROGRAM_BUILD_LOG, 
          0, 
          NULL, 
          &log_size
        );

        std::vector<char> log(log_size);
        
        // Get the log
        clGetProgramBuildInfo(
          object_, 
          d, 
          CL_PROGRAM_BUILD_LOG, 
          log_size, 
          log.data(),
          nullptr
        );
        
        // Print the log
        std::cerr << log.data() << std::endl;
        throw std::runtime_error("Failed to build program");
      }

      /*
      if (options.size() != 0) {
        std::string compile_opts("");
        for (auto s : options) {
          compile_opts += s + std::string(" ");
        }
        CL_CHECK_ERROR(
          clBuildProgram(
            object_,
            1,
            &d,
            compile_opts.c_str(),
            nullptr,
            nullptr
          )
        );
      } else {
        CL_CHECK_ERROR(
          clBuildProgram(
            object_,
            1,
            &d,
            nullptr,
            nullptr,
            nullptr
          )
        );
      }
      */
    }
  }; // class Program

} // namespace cl


  
#pragma once

#include "ocl/common.h"


namespace ocl {

  class Kernel : public ObjectBase<cl_kernel> {
  public:
    explicit Kernel(Program& program, const std::string& kernel_name) {
      cl_int status = 0;
      object_ = clCreateKernel(program(), kernel_name.data(), &status);
      if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create kernel.");
      }
    }
    
    virtual ~Kernel() {
      
    }

    void Run(CommandQueue& queue, 
             cl_uint work_dim, 
             const size_t *global_work_offset, 
             const size_t *global_work_size, 
             const size_t *local_work_size, 
             cl_uint num_events_in_wait_list=0, 
             const cl_event *event_wait_list=nullptr, 
             cl_event *event=nullptr) {
      CL_CHECK_ERROR(clEnqueueNDRangeKernel(
          queue(), 
          object_, 
          work_dim, 
          global_work_offset, 
          global_work_size,
          local_work_size,
          num_events_in_wait_list,
          event_wait_list,
          event
        )
      );
    }

    // Sets all arguments in one go using parameter packs. 
    // Note that this overwrites previously set
    // arguments using 'SetArgument' or 'SetArguments'.
    template <typename... Args>
    void SetArguments(Args&... args) {
      SetArgumentsRecursive(0, args...);
    }


  private:

    template <typename T>
    void SetArgument(const size_t index, T& arg) {
      CL_CHECK_ERROR(
        clSetKernelArg(
          object_,
          static_cast<cl_uint>(index),
          sizeof(T),
          static_cast<const void*>(&arg)
        )
      );
    }

    template <typename T>
    void SetArgument(const size_t index, Buffer<T>& arg) {
      CL_CHECK_ERROR(
        clSetKernelArg(
          object_,
          static_cast<cl_uint>(index),
          sizeof(T),
          static_cast<const void*>(&arg())
        )
      );
    }

    template <typename T> 
    void SetArgumentsRecursive(const size_t index, T& first) {
      SetArgument(index, first);
    }

    template <typename T, typename... Args> 
    void SetArgumentsRecursive(const size_t index, T& first, Args&... args) {
      SetArgument(index, first);
      SetArgumentsRecursive(index+1, args...);
    }


  }; // class Kernel

} // namespace cl


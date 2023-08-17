#pragma once

#include "ocl/common.h"

namespace ocl {

  class CommandQueue : public ObjectBase<cl_command_queue> {
  public:
    explicit CommandQueue(const Context& context, const Device& device) {
      cl_int status = 0;
      object_ = clCreateCommandQueue(
        context(), device(), CL_QUEUE_PROFILING_ENABLE, &status
      );
      if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create command queue.");
      }
    }

    virtual ~CommandQueue() {}

    void Finish() const {
      clFinish(object_);
    }
  }; // class CommandQueue

} // namespace cl

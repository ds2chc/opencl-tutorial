#pragma once

#include "ocl/common.h"

namespace ocl {

  /// @brief C++ 11 supported cl_context
  class Context : public ObjectBase<cl_context> {
  public:
  
    explicit Context(const Device& device) {
      cl_int status = 0;
      cl_platform_id platform = device.PlatformID();
      cl_context_properties properties[] = {
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 
        0
      };
      
      object_ = clCreateContext(
        properties,
        1,
        &device(),
        nullptr,
        nullptr,
        &status
      );
      if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create context with status: " + std::to_string(status) + "." );
      }
    }

    virtual ~Context() {
      
    }

  }; // class Context

} // namespace cl
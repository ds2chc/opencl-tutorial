#pragma once

#include "ocl/common.h"


namespace ocl {

  template <typename T>
  class Buffer : public ObjectBase<cl_mem> {
  public:
    explicit Buffer(const Context& context, size_t num_elmts) {
      cl_int status = 0;
      object_ = clCreateBuffer(
        context(),
        CL_MEM_READ_WRITE,
        num_elmts * sizeof(T),
        nullptr,
        &status
      );
      if (status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create memory.");
      }
    }

    virtual ~Buffer() {

    }

    void CopyFromHost(CommandQueue& queue,
                     const T* src, 
                     size_t num_elmts, 
                     size_t offset=0,
                     bool blocking=true,
                     cl_uint num_events_in_wait_list=0,
                     const cl_event* event_wait_list=nullptr,
                     cl_event* event=nullptr) {
      CL_CHECK_ERROR(
        clEnqueueWriteBuffer(
          queue(), 
          object_, 
          blocking ? CL_TRUE : CL_FALSE, 
          offset, 
          num_elmts * sizeof(T), 
          static_cast<const void*>(src),
          num_events_in_wait_list,
          event_wait_list,
          event
        )
      );
    }

    void CopyFromDevice(CommandQueue& queue,
                        T* dst, 
                        size_t num_elmts, 
                        size_t offset=0,
                        bool blocking=true,
                        cl_uint num_events_in_wait_list=0,
                        const cl_event* event_wait_list=nullptr,
                        cl_event* event=nullptr) {
      CL_CHECK_ERROR(
        clEnqueueReadBuffer(
          queue(), 
          object_, 
          blocking ? CL_TRUE : CL_FALSE, 
          offset, 
          num_elmts * sizeof(T), 
          static_cast<void*>(dst),
          num_events_in_wait_list,
          event_wait_list,
          event
        )
      );
    }
  }; // class Buffer

} // namespace cl

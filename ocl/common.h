#pragma once

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <string>
#include <system_error>
#include <vector>
#include <memory>

#include <CL/cl.h>



#define CL_CHECK_ERROR(_call) do {      \
  int _err = (_call);                   \
  if (_err != CL_SUCCESS) {             \
    throw std::runtime_error(           \
      "(error code: "                   \
      + std::to_string(_err) + ") "     \
      #_call                            \
    );                                  \
  }                                     \
} while (0)



namespace ocl {

  template <typename T>
  struct RawObjectHandler {
    static void Release(T type) {

    }
  };

  template <typename _Object>
  class ObjectBase {
  public:
    /// @brief Default constructor
    ObjectBase() : object_(nullptr) {};

    /// @brief Constructor based on the regluar OpenCL data type
    ObjectBase(const _Object obj) : object_(obj) {}

    ObjectBase(const ObjectBase& rhs) : object_(rhs.object_) {

    }

    ObjectBase(ObjectBase&& rhs) : object_(std::move(rhs.object_)) {
      
    }

    virtual ~ObjectBase() {
      Release();
    }

    /// @brief Copy assignment
    ObjectBase& operator= (const ObjectBase& rhs) {
      if (this->object_ != rhs.object_) {
        Release();
        this->object_ = rhs.object_;
      }
      return *this;
    }

    /// @brief Move assignment
    ObjectBase& operator= (ObjectBase&& rhs) {
      if (this->object_ != rhs.object_) {
        Release();
        this->object_ = rhs.object_;
      }
      return *this;
    }

    /// @brief Get raw object type
    const _Object& operator() () const { return object_; }


  protected:
    _Object object_;

    void Release() {
      if (object_ != nullptr) {
        RawObjectHandler<_Object>::Release(object_);
        object_ = nullptr;
      }
    }

  }; // class ObjectBase


  template <> struct RawObjectHandler<cl_device_id> {
    static void Release(cl_device_id device) {
      CL_CHECK_ERROR(clReleaseDevice(device));
    } 
  };

  template <> struct RawObjectHandler<cl_context> {
    static void Release(cl_context context) {
      CL_CHECK_ERROR(clReleaseContext(context));
    } 
  };
  
  template <> struct RawObjectHandler<cl_program> {
    static void Release(cl_program program) {
      CL_CHECK_ERROR(clReleaseProgram(program));
    }
  };

  template <> struct RawObjectHandler<cl_kernel> {
    static void Release(cl_kernel kernel) {
      CL_CHECK_ERROR(clReleaseKernel(kernel));
    }
  };

  template <> struct RawObjectHandler<cl_mem> {
    static void Release(cl_mem mem) {
      CL_CHECK_ERROR(clReleaseMemObject(mem));
    }
  };

} // namespace cl
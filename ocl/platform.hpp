#pragma once

#include "ocl/common.h"


namespace ocl {

  class Platform : public ObjectBase<cl_platform_id> {
  public:
    typedef ObjectBase<cl_platform_id> Base;

    /// @brief Default constructor
    Platform() : Base() {}

    /// @brief Initializes the platform with given index
    /// @param platform_id Index of platform to use, must be smaller than number of available platforms.
    explicit Platform(const size_t platform_id) {
      cl_uint num_platforms = 0;
      CL_CHECK_ERROR(clGetPlatformIDs(0, nullptr, &num_platforms));
      if (num_platforms == 0) {
        throw std::runtime_error("Platform: no platforms found.");
      }
      if (platform_id >= num_platforms) {
        throw std::runtime_error("Platform: invalid platform id " + std::to_string(platform_id));
      }

      auto platforms = std::vector<cl_platform_id>(num_platforms);
      CL_CHECK_ERROR(clGetPlatformIDs(num_platforms, platforms.data(), nullptr));
      object_ = platforms[platform_id];
    }

    virtual ~Platform() {
      
    }

    /// @brief Get the number of total devices in this platform.
    size_t GetNumDevices() const {
      cl_uint num_devices = 0;
      CL_CHECK_ERROR(clGetDeviceIDs(object_, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices));
      return static_cast<size_t>(num_devices);
    }

    // Methods to retrieve platform information
    std::string GetName() const { return GetInfoString(CL_PLATFORM_NAME); }
    std::string GetVendorName() const { return GetInfoString(CL_PLATFORM_VENDOR); }
    std::string GetVersion() const { return GetInfoString(CL_PLATFORM_VERSION); }

  private:

    // Private helper functions
    std::string GetInfoString(const cl_device_info info) const {
      size_t bytes = 0;
      CL_CHECK_ERROR(clGetPlatformInfo(object_, info, 0, nullptr, &bytes));
      auto result = std::string{};
      result.resize(bytes);
      CL_CHECK_ERROR(clGetPlatformInfo(object_, info, bytes, &result[0], nullptr));
      result.resize(strlen(result.c_str())); // Removes any trailing '\0'-characters
      return result;
    }

  }; // class Platform



  // Retrieves a vector with all platforms
  inline std::vector<Platform> GetAllPlatforms() {
    cl_uint num_platforms = 0;
    CL_CHECK_ERROR(clGetPlatformIDs(0, nullptr, &num_platforms));

    auto all_platforms = std::vector<Platform>();
    for (cl_uint i = 0; i < num_platforms; i++) {
      all_platforms.push_back(Platform(i));
    }
    
    return all_platforms;
  }

} // namespace cl
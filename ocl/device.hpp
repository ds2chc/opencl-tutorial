#pragma once

#include "ocl/common.h"


namespace ocl {

  // C++11 version of 'cl_device_id'
  class Device : public ObjectBase<cl_device_id> {
  public:
    typedef ObjectBase<cl_device_id> Base;
    
    Device() : Base() {}

    // Initialize the device. Note that this constructor can throw exceptions!
    explicit Device(const Platform &platform, const size_t device_id) {
      auto num_devices = platform.GetNumDevices();
      if (num_devices == 0) {
        throw std::runtime_error("Device: no devices found");
      }
      if (device_id >= num_devices) {
        throw std::runtime_error("Device: invalid device ID "+std::to_string(device_id));
      }

      auto devices = std::vector<cl_device_id>(num_devices);
      CL_CHECK_ERROR(clGetDeviceIDs(platform(), CL_DEVICE_TYPE_ALL, static_cast<cl_uint>(num_devices),
                                devices.data(), nullptr));
      object_ = devices[device_id];
      clRetainDevice(object_);
      for (size_t n = 0; n < num_devices; n++) {
        clReleaseDevice(devices[n]);
        devices[n] = nullptr;
      }
    }

    virtual ~Device() {

    }

    // Methods to retrieve device information
    cl_platform_id PlatformID() const { return GetInfo<cl_platform_id>(CL_DEVICE_PLATFORM); }

    std::string Version() const { return GetInfoString(CL_DEVICE_VERSION); }

    size_t VersionNumber() const
    {
      std::string version_string = Version().substr(7);
      // Space separates the end of the OpenCL version number from the beginning of the
      // vendor-specific information.
      size_t next_whitespace = version_string.find(' ');
      size_t version = (size_t) (100.0 * std::stod(version_string.substr(0, next_whitespace)));
      return version;
    }
    std::string Vendor() const { return GetInfoString(CL_DEVICE_VENDOR); }
    std::string Name() const { return GetInfoString(CL_DEVICE_NAME); }
    std::string Type() const {
      auto type = GetInfo<cl_device_type>(CL_DEVICE_TYPE);
      switch(type) {
        case CL_DEVICE_TYPE_CPU: return "CPU";
        case CL_DEVICE_TYPE_GPU: return "GPU";
        case CL_DEVICE_TYPE_ACCELERATOR: return "accelerator";
        default: return "default";
      }
    }
    size_t MaxWorkGroupSize() const { return GetInfo<size_t>(CL_DEVICE_MAX_WORK_GROUP_SIZE); }
    size_t MaxWorkItemDimensions() const {
      return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS));
    }
    std::vector<size_t> MaxWorkItemSizes() const {
      return GetInfoVector<size_t>(CL_DEVICE_MAX_WORK_ITEM_SIZES);
    }
    unsigned long LocalMemSize() const {
      return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE));
    }

    std::string Capabilities() const { return GetInfoString(CL_DEVICE_EXTENSIONS); }
    bool HasExtension(const std::string &extension) const {
      const auto extensions = Capabilities();
      return extensions.find(extension) != std::string::npos;
    }
    bool SupportsFP64() const {
      return HasExtension("cl_khr_fp64");
    }
    bool SupportsFP16() const {
      if (Name() == "Mali-T628") { return true; } // supports fp16 but not cl_khr_fp16 officially
      return HasExtension("cl_khr_fp16");
    }

    size_t CoreClock() const {
      return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_CLOCK_FREQUENCY));
    }
    size_t ComputeUnits() const {
      return static_cast<size_t>(GetInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS));
    }
    unsigned long MemorySize() const {
      return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE));
    }
    unsigned long MaxAllocSize() const {
      return static_cast<unsigned long>(GetInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE));
    }
    size_t MemoryClock() const { return 0; } // Not exposed in OpenCL
    size_t MemoryBusWidth() const { return 0; } // Not exposed in OpenCL

    // Configuration-validity checks
    bool IsLocalMemoryValid(const cl_ulong local_mem_usage) const {
      return (local_mem_usage <= LocalMemSize());
    }
    bool IsThreadConfigValid(const std::vector<size_t> &local) const {
      auto local_size = size_t{1};
      for (const auto &item: local) { local_size *= item; }
      for (auto i=size_t{0}; i<local.size(); ++i) {
        if (local[i] > MaxWorkItemSizes()[i]) { return false; }
      }
      if (local_size > MaxWorkGroupSize()) { return false; }
      if (local.size() > MaxWorkItemDimensions()) { return false; }
      return true;
    }
  /*
    // Query for a specific type of device or brand
    bool IsCPU() const { return Type() == "CPU"; }
    bool IsGPU() const { return Type() == "GPU"; }
    bool IsAMD() const { return Vendor() == "AMD" ||
                                Vendor() == "Advanced Micro Devices, Inc." ||
                                Vendor() == "AuthenticAMD"; }
    bool IsNVIDIA() const { return Vendor() == "NVIDIA" ||
                                  Vendor() == "NVIDIA Corporation"; }
    bool IsIntel() const { return Vendor() == "INTEL" ||
                                  Vendor() == "Intel" ||
                                  Vendor() == "GenuineIntel" ||
                                  Vendor() == "Intel(R) Corporation"; }
    bool IsARM() const { return Vendor() == "ARM"; }
    bool IsQualcomm() const { return Vendor() == "QUALCOMM"; }

    // Platform specific extensions
    std::string AMDBoardName() const { // check for 'cl_amd_device_attribute_query' first
      #ifndef CL_DEVICE_BOARD_NAME_AMD
        #define CL_DEVICE_BOARD_NAME_AMD 0x4038
      #endif
      return GetInfoString(CL_DEVICE_BOARD_NAME_AMD);
    }
    std::string NVIDIAComputeCapability() const { // check for 'cl_nv_device_attribute_query' first
      #ifndef CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV
        #define CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV 0x4000
      #endif
      #ifndef CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV
        #define CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV 0x4001
      #endif
      return std::string{"SM"} + std::to_string(GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV)) +
            std::string{"."} + std::to_string(GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MINOR_NV));
    }

    // Returns if the Nvidia chip is a Volta or later archicture (sm_70 or higher)
    bool IsPostNVIDIAVolta() const {
      if(HasExtension("cl_nv_device_attribute_query")) {
        return GetInfo<cl_uint>(CL_DEVICE_COMPUTE_CAPABILITY_MAJOR_NV) >= 7;
      }
      return false;
    }

    // Returns the Qualcomm Adreno GPU version (i.e. a650, a730, a740, etc.)
    std::string AdrenoVersion() const {
      if (IsQualcomm()) {
        return GetInfoString(CL_DEVICE_OPENCL_C_VERSION);
      }
      else { return std::string{""}; }
    }

    // Retrieves the above extra information (if present)
    std::string GetExtraInfo() const {
      if (HasExtension("cl_amd_device_attribute_query")) { return AMDBoardName(); }
      if (HasExtension("cl_nv_device_attribute_query")) { return NVIDIAComputeCapability(); }
      else { return std::string{""}; }
    }
  */

  private:
    // Private helper functions
    template <typename T>
    T GetInfo(const cl_device_info info) const {
      auto bytes = size_t{0};
      CL_CHECK_ERROR(clGetDeviceInfo(object_, info, 0, nullptr, &bytes));
      auto result = T(0);
      CL_CHECK_ERROR(clGetDeviceInfo(object_, info, bytes, &result, nullptr));
      return result;
    }
    template <typename T>
    std::vector<T> GetInfoVector(const cl_device_info info) const {
      auto bytes = size_t{0};
      CL_CHECK_ERROR(clGetDeviceInfo(object_, info, 0, nullptr, &bytes));
      auto result = std::vector<T>(bytes/sizeof(T));
      CL_CHECK_ERROR(clGetDeviceInfo(object_, info, bytes, result.data(), nullptr));
      return result;
    }
    std::string GetInfoString(const cl_device_info info) const {
      auto bytes = size_t{0};
      CL_CHECK_ERROR(clGetDeviceInfo(object_, info, 0, nullptr, &bytes));
      auto result = std::string{};
      result.resize(bytes);
      CL_CHECK_ERROR(clGetDeviceInfo(object_, info, bytes, &result[0], nullptr));
      result.resize(strlen(result.c_str())); // Removes any trailing '\0'-characters
      return result;
    }

  };

} // namespace cl
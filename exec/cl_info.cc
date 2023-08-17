#include <stdio.h>
#include <stdlib.h>

#include <iostream>
#include <vector>
#include <string>

#include "ocl/platform.hpp"
#include "ocl/device.hpp"

int main(void) {
  std::vector<ocl::Platform> all_platforms = ocl::GetAllPlatforms();
  size_t num_platforms = all_platforms.size();
  
  for (size_t n = 0; n < all_platforms.size(); n++) {
    std::cout << "Platform " << n << ": " << all_platforms[n].GetName() << std::endl;

    size_t num_devices = all_platforms[n].GetNumDevices();
    for (size_t m = 0; m < num_devices; m++) {
      ocl::Device device(all_platforms[n], m);
      std::cout << "Device " << m << ": " << device.Name() << std::endl;
      std::cout << "Number of compute units: " << device.ComputeUnits() << std::endl;
      std::cout << "Max alloc. size: " << device.MaxAllocSize() << std::endl;
      std::cout << "Local memory size: " << device.LocalMemSize() << std::endl;
      std::cout << "Max work group size: " << device.MaxWorkGroupSize() << std::endl;
      std::cout << "Max work item dimensions: " << device.MaxWorkItemDimensions() << std::endl;

      auto max_work_item_sizes = device.MaxWorkItemSizes();
      std::cout << "Max work item sizes: (";
      for (size_t i = 0; i < max_work_item_sizes.size(); i++) {
        std::cout << max_work_item_sizes[i];
        if (i != max_work_item_sizes.size() -1) {
          std::cout << ", ";
        }
      }
      std::cout << ")" << std::endl;
    }

    std::cout << "-------------------------------" << std::endl;
  }
  
  return 0;
}
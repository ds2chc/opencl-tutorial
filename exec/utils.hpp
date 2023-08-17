#pragma once

#include <string>
#include <fstream>

namespace utils {

  std::string ReadKernelFileFromDisk(const std::string& file_path) {
    // Get size of file
    std::ifstream input_file_stream(file_path);
    input_file_stream.seekg(0, std::ios::end);
    size_t size = input_file_stream.tellg();
    std::string buffer(size, ' ');

    // Read input file
    input_file_stream.seekg(0);
    input_file_stream.read(&buffer[0], size);
    return buffer; 
  }

} // namespace utils
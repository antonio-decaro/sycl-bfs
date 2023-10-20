#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <array>
#include "kernel_sizes.hpp"
#include "host_data.hpp"
#include "sycl_data.hpp"
#include "types.hpp"

namespace s = sycl;

void bottomUpBFS(s::queue& q, SYCL_CompressedGraphData& data) {
  q.submit([&](s::handler& cgh) {    

  }).wait();
}

class BottomUpBFS {
public:
  BottomUpBFS(std::vector<CSRHostData>& data) : 
    data(data)
  {}

  template<size_t sg_size, bool init = false>
  void run(const size_t local_size = DEFAULT_WORK_GROUP_SIZE) {
    CompressedHostData compressed_data(data);
    SYCL_CompressedGraphData sycl_data(compressed_data);

    
    sycl_data.write_back();

  }
private:
  std::vector<CSRHostData>& data;
};

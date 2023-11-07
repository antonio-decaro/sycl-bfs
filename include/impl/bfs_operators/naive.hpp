#ifndef __NAIVE_HPP__
#define __NAIVE_HPP__

#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include "host_data.hpp"
#include "kernel_sizes.hpp"
#include "sycl_data.hpp"
#include "impl/simpl_bfs.hpp"

namespace s = sycl;

class NaiveBFSOperator : public SingleBFSOperator {
public:
  void operator() (sycl::queue& queue, SYCL_SimpleGraphData& data, std::vector<sycl::event>& events) {
    bool *changed = s::malloc_shared<bool>(1, queue);
    queue.fill(changed, false, 1);
    int level = 0;
    s::buffer<int, 1> distances{data.num_nodes};
    queue.submit([&](s::handler& h) {
      s::accessor distances_acc(distances, h, s::write_only, s::no_init);
      h.parallel_for(s::range<1>{data.num_nodes}, [=](s::id<1> idx) {
        if (idx[0] == 0) {
          distances_acc[0] = 0;
        } else {
          distances_acc[idx[0]] = -1;
        }
      });
    }).wait();

    do {
      *changed = false;
      auto e = queue.submit([&](sycl::handler &h) {
        s::accessor offsets_acc(data.edges_offsets, h, s::read_only);
        s::accessor edges_acc(data.edges, h, s::read_only);
        s::accessor parents_acc(data.parents, h, s::write_only, s::no_init);
        s::accessor distances_acc(distances, h, s::read_write);

        h.parallel_for(s::range<1>{data.num_nodes}, [=, num_nodes=data.num_nodes](s::id<1> idx) {
            int node = idx[0];
            if (distances_acc[node] == level) {
                for (int i = offsets_acc[node]; i < offsets_acc[node + 1]; i++) {
                    int neighbor = edges_acc[i];
                    if (distances_acc[neighbor] == -1) {
                        distances_acc[neighbor] = level + 1;
                        parents_acc[neighbor] = node;
                        *changed = true;
                    }
                }
            }
        }); });
      events.push_back(e);
      e.wait();
      level++;
    } while (*changed);

    std::cout << "[*] Max depth reached: " << level << std::endl;
  }
};

#endif
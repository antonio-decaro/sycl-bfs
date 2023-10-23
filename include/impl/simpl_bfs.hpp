#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "host_data.hpp"
#include "kernel_sizes.hpp"
#include "sycl_data.hpp"

namespace s = sycl;

void dummy_kernel(sycl::queue& queue, SYCL_SimpleGraphData& data) {

    // dummy kernel to init data
    queue.submit([&](sycl::handler& h) {
        s::accessor offsets_acc(data.edges_offsets, h, s::read_only);
        s::accessor edges_acc(data.edges, h, s::read_only);
        s::accessor distances_acc(data.distances, h, s::read_write, s::no_init);
        s::accessor parents_acc(data.parents, h, s::write_only, s::no_init);

        h.parallel_for(s::range<1>{data.num_nodes}, [=](s::id<1> idx) {
            auto a = offsets_acc[idx[0]];
            if (idx[0] == 0) {
                distances_acc[idx[0]] = 0;
                parents_acc[idx[0]] = -1;
            }
            else {
                distances_acc[idx[0]] = -1;
                parents_acc[idx[0]] = -1;
            }
        });
    }).wait();
}

template<size_t sg_size>
void frontier_BFS(sycl::queue& queue, SYCL_SimpleGraphData& data, std::vector<sycl::event>& events) {

    int* frontier = s::malloc_device<int>(data.num_nodes, queue);
    int* frontier_size = s::malloc_shared<int>(1, queue);
    int* old_frontier_size = s::malloc_shared<int>(1, queue);
    queue.fill(frontier, 0, data.num_nodes).wait(); // init the frontier with the the node 0
    queue.fill(frontier_size, 0, 1).wait();
    queue.fill(old_frontier_size, 1, 1).wait();

    int level = 0;
    while (*old_frontier_size) {
        auto e = queue.submit([&](s::handler& h) [[intel::reqd_sub_group_size(sg_size)]] {
            s::accessor offsets_acc(data.edges_offsets, h, s::read_only);
            s::accessor edges_acc(data.edges, h, s::read_only);
            s::accessor distances_acc(data.distances, h, s::read_write);
            s::accessor parents_acc(data.parents, h, s::write_only, s::no_init);

            size_t size = *old_frontier_size;
            h.parallel_for(s::range<1>{size}, [=](s::id<1> idx) {
                s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::device> frontier_size_ref(*frontier_size);
                int node = frontier[idx[0]];
                
                for (int i = offsets_acc[node]; i < offsets_acc[node + 1]; i++) {
                    int neighbor = edges_acc[i];
                    if (distances_acc[neighbor] == -1) {
                        int pos = frontier_size_ref.fetch_add(1);
                        distances_acc[neighbor] = distances_acc[node] + 1;
                        parents_acc[neighbor] = node;
                        frontier[pos] = neighbor;
                    }
                }

            });
            
        });
        events.push_back(e);
        e.wait();
        *old_frontier_size = *frontier_size;
        *frontier_size = 0;
    }

    s::free(frontier, queue);
    s::free(frontier_size, queue);
    s::free(old_frontier_size, queue);
}

void multi_events_BFS(sycl::queue& queue, SYCL_SimpleGraphData& data, std::vector<sycl::event>& events) {

    bool* changed = s::malloc_shared<bool>(1, queue);
    queue.fill(changed, false, 1);
    int level = 0;

    do {
        *changed = false;
        auto e = queue.submit([&](sycl::handler& h) {
            s::accessor offsets_acc(data.edges_offsets, h, s::read_only);
            s::accessor edges_acc(data.edges, h, s::read_only);
            s::accessor distances_acc(data.distances, h, s::read_write);
            s::accessor parents_acc(data.parents, h, s::write_only, s::no_init);

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
            });
        });
        events.push_back(e);
        e.wait();
        level++;
    } while(*changed);

    std::cout << "[*] Max depth reached: " << level << std::endl;
}

class SimpleBFS {
private:
    CSRHostData& data;

public:
    SimpleBFS(CSRHostData& data) : 
        data(data) {}

    void run() {
        // SYCL queue definition
      sycl::queue queue (sycl::gpu_selector_v, 
                        sycl::property_list{sycl::property::queue::enable_profiling{}});

      SYCL_SimpleGraphData sycl_data(data);

      std::vector<sycl::event> events;

      // dummy_kernel(queue, sycl_data);

      auto start_glob = std::chrono::high_resolution_clock::now();
      // multi_events_BFS(queue, sycl_data, events);
      frontier_BFS<32>(queue, sycl_data, events);
      auto end_glob = std::chrono::high_resolution_clock::now();

      long duration = 0;
      for (sycl::event& e : events) {
          auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
          auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
          duration += (end - start);
      }

      sycl_data.write_back();

      std::cout << "[*] Kernels duration: " << duration / 1000 << " us" << std::endl;
      std::cout << "[*] Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count() << " us" << std::endl;
    }
};
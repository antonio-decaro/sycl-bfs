#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "sycl_data.hpp"
#include "kernel_sizes.hpp"
#include "bfs.hpp"


namespace s = sycl;

void copy_data(SYCL_GraphData& sycl_data, HostData& data) {
    auto dist_acc = sycl_data.distances.get_host_access();
    auto parent_acc = sycl_data.parents.get_host_access();

    for (int i = 0; i < data.num_nodes; i++) {
        data.distances[i] = dist_acc[i];
        data.parents[i] = parent_acc[i];
    }
}

void sm_BFS(sycl::queue& queue, SYCL_GraphData& data, std::vector<sycl::event>& events) {

    bool* changed = s::malloc_shared<bool>(1, queue);
    queue.fill(changed, false, 1);
    int level = 0;

    do {
        *changed = false;
        auto e = queue.submit([&](sycl::handler& h) {
            s::stream os{128, 16, h};
            auto offsets_acc = data.offsets.get_access<s::access::mode::read>(h);
            auto edges_acc = data.edges.get_access<s::access::mode::read>(h);
            auto distances_acc = data.distances.get_access<s::access::mode::read_write>(h);
            auto parents_acc = data.parents.get_access<s::access::mode::read_write>(h);

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
}

void SimpleBFS::run() {

    // SYCL queue definition
    sycl::queue queue (sycl::gpu_selector_v, 
                       sycl::property_list{sycl::property::queue::enable_profiling{}});

    SYCL_GraphData sycl_data(data);

    std::vector<sycl::event> events;

    auto start_glob = std::chrono::high_resolution_clock::now();
    sm_BFS(queue, sycl_data, events);
    auto end_glob = std::chrono::high_resolution_clock::now();

    long duration = 0;
    for (sycl::event& e : events) {
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        duration += (end - start);
    }

    std::cout << "[*] Kernels duration: " << duration / 1000 << " us" << std::endl;
    std::cout << "[*] Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count() << " us" << std::endl;

    copy_data(sycl_data, data);
}
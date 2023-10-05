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

void write_back(SYCL_GraphData& sycl_data) {
    sycl_data.distances.set_final_data(sycl_data.host_data.distances.data());
    sycl_data.parents.set_final_data(sycl_data.host_data.parents.data());
    sycl_data.distances.set_write_back(true);
    sycl_data.parents.set_write_back(true);
}

void dummy_kernel(sycl::queue& queue, SYCL_GraphData& data) {

    // dummy kernel to init data
    queue.submit([&](sycl::handler& h) {
        auto offsets_acc = data.offsets.get_access<s::access::mode::read>(h);
        auto edges_acc = data.edges.get_access<s::access::mode::read>(h);
        auto distances_acc = data.distances.get_access<s::access::mode::discard_write>(h);
        auto parents_acc = data.parents.get_access<s::access::mode::discard_write>(h);

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

void frontier_BFS(sycl::queue& queue, SYCL_GraphData& data, std::vector<sycl::event>& events) {

    int* frontier = s::malloc_device<int>(data.num_nodes, queue);
    int* frontier_size = s::malloc_shared<int>(1, queue);
    int* old_frontier_size = s::malloc_shared<int>(1, queue);
    queue.fill(frontier, 0, data.num_nodes).wait(); // init the frontier with the the node 0
    queue.fill(frontier_size, 0, 1).wait();
    queue.fill(old_frontier_size, 1, 1).wait();

    int level = 0;
    while (*old_frontier_size) {
        auto e = queue.submit([&](s::handler& h) {
            auto offsets_acc = data.offsets.get_access<s::access::mode::read>(h);
            auto edges_acc = data.edges.get_access<s::access::mode::read>(h);
            auto distances_acc = data.distances.get_access<s::access::mode::read_write>(h);
            auto parents_acc = data.parents.get_access<s::access::mode::discard_write>(h);

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

void multi_events_BFS(sycl::queue& queue, SYCL_GraphData& data, std::vector<sycl::event>& events) {

    bool* changed = s::malloc_shared<bool>(1, queue);
    queue.fill(changed, false, 1);
    int level = 0;

    do {
        *changed = false;
        auto e = queue.submit([&](sycl::handler& h) {
            auto offsets_acc = data.offsets.get_access<s::access::mode::read>(h);
            auto edges_acc = data.edges.get_access<s::access::mode::read>(h);
            auto distances_acc = data.distances.get_access<s::access::mode::read_write>(h);
            auto parents_acc = data.parents.get_access<s::access::mode::discard_write>(h);

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

void SimpleBFS::run() {

    // SYCL queue definition
    sycl::queue queue (sycl::gpu_selector_v, 
                       sycl::property_list{sycl::property::queue::enable_profiling{}});

    SYCL_GraphData sycl_data(data);

    std::vector<sycl::event> events;

    dummy_kernel(queue, sycl_data);

    auto start_glob = std::chrono::high_resolution_clock::now();
    // multi_events_BFS(queue, sycl_data, events);
    frontier_BFS(queue, sycl_data, events);
    auto end_glob = std::chrono::high_resolution_clock::now();

    long duration = 0;
    for (sycl::event& e : events) {
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        duration += (end - start);
    }

    write_back(sycl_data); 

    std::cout << "[*] Kernels duration: " << duration / 1000 << " us" << std::endl;
    std::cout << "[*] Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count() << " us" << std::endl;
}
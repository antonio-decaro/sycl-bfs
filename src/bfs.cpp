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

[[deprecated("This function is not required anymore to copy data back to the host.")]]
void copy_data(SYCL_GraphData& sycl_data, HostData& data) {
    auto dist_acc = sycl_data.distances.get_host_access();
    auto parent_acc = sycl_data.parents.get_host_access();

    for (int i = 0; i < data.num_nodes; i++) {
        data.distances[i] = dist_acc[i];
        data.parents[i] = parent_acc[i];
    }
}

// TODO to fix
void single_queue_BFS(sycl::queue& queue, SYCL_GraphData& data, std::vector<sycl::event>& events) {

    int* changed = s::malloc_device<int>(1, queue);
    queue.fill(changed, false, 1);
    int* level = s::malloc_shared<int>(1, queue);
    queue.fill(level, 0, 1);

    s::range<1> global{data.num_nodes + (WORK_GROUP_SIZE - (data.num_nodes % WORK_GROUP_SIZE))};
    s::range<1> local{WORK_GROUP_SIZE};

    {
        auto e = queue.submit([&](sycl::handler& h) {
            auto offsets_acc = data.offsets.get_access<s::access::mode::read>(h);
            auto edges_acc = data.edges.get_access<s::access::mode::read>(h);
            auto distances_acc = data.distances.get_access<s::access::mode::read_write>(h);
            auto parents_acc = data.parents.get_access<s::access::mode::discard_write>(h);

            s::stream out(256, 16, h);

            h.parallel_for(s::nd_range<1>{global, local}, [=, num_nodes=data.num_nodes](s::nd_item<1> idx) {
                
                s::atomic_ref<int, s::memory_order::acq_rel, s::memory_scope::device, s::access::address_space::global_space> changed_ref {*changed};

                out << "Init\n";
                do {
                    int node = idx.get_global_linear_id();
                    changed_ref.store(0);
                    if (node < num_nodes && distances_acc[node] == *level) {
                        for (int i = offsets_acc[node]; i < offsets_acc[node + 1]; i++) {
                            int neighbor = edges_acc[i];
                            if (distances_acc[neighbor] == -1) {
                                distances_acc[neighbor] = *level + 1;
                                parents_acc[neighbor] = node;
                                changed_ref.store(1);
                            }
                        }
                    }
                    out << "Before\n";

                    idx.barrier(s::access::fence_space::global_space);
                    if (node == 0) {
                        (*level)++;
                    }
                    idx.barrier(s::access::fence_space::global_space);
                    out << "After\n";
                } while (changed_ref.load());
                out << "I'm out " << idx.get_global_linear_id() << "\n";
            });
        });
        events.push_back(e);
        e.wait();
    }

    std::cout << "[*] Max depth reached: " << *level << std::endl;
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

    auto start_glob = std::chrono::high_resolution_clock::now();
    multi_events_BFS(queue, sycl_data, events);
    // single_queue_BFS(queue, sycl_data, events);
    auto end_glob = std::chrono::high_resolution_clock::now();

    long duration = 0;
    for (sycl::event& e : events) {
        auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
        duration += (end - start);
    }

    copy_data(sycl_data, data); 

    std::cout << "[*] Kernels duration: " << duration / 1000 << " us" << std::endl;
    std::cout << "[*] Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count() << " us" << std::endl;
}
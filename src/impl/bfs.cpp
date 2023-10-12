#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "kernel_sizes.hpp"
#include "bfs.hpp"


namespace s = sycl;

class SYCL_GraphData {
public:
    SYCL_GraphData(CSRHostData& data) :
        host_data(data),
        num_nodes(data.num_nodes),
        edges_offsets(sycl::buffer<size_t, 1>{data.csr.offsets.data(), sycl::range{data.csr.offsets.size()}}),
        edges(sycl::buffer<nodeid_t, 1>{data.csr.edges.data(), sycl::range{data.csr.edges.size()}}),
        distances(sycl::buffer<distance_t, 1>{data.distances.data(), sycl::range{data.distances.size()}}),
        parents(sycl::buffer<nodeid_t, 1>{data.parents.data(), sycl::range{data.parents.size()}})
    {
        distances.set_write_back(false);
        parents.set_write_back(false);
    }

    void write_back() {
        distances.set_final_data(host_data.distances.data());
        parents.set_final_data(host_data.parents.data());
        distances.set_write_back(true);
        parents.set_write_back(true);
    }

    size_t num_nodes;
    CSRHostData& host_data;
    sycl::buffer<nodeid_t, 1> parents, edges;
    sycl::buffer<size_t, 1> edges_offsets;
    sycl::buffer<distance_t, 1> distances;
};

void dummy_kernel(sycl::queue& queue, SYCL_GraphData& data) {

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

void multi_frontier_BFS(sycl::queue& queue, SYCL_GraphData& data, std::vector<sycl::event>& events) {

    int* frontier = s::malloc_device<int>(data.num_nodes, queue);
    int* frontier_size = s::malloc_shared<int>(1, queue);
    int* old_frontier_size = s::malloc_shared<int>(1, queue);
    queue.fill(frontier, 0, data.num_nodes).wait(); // init the frontier with the the node 0
    queue.fill(frontier_size, 0, 1).wait();
    queue.fill(old_frontier_size, 1, 1).wait();

    int level = 0;
    while (*old_frontier_size) {
        auto e = queue.submit([&](s::handler& h) {
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

void multi_events_BFS(sycl::queue& queue, SYCL_GraphData& data, std::vector<sycl::event>& events) {

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

void SimpleBFS::run() {

    // SYCL queue definition
    sycl::queue queue (sycl::gpu_selector_v, 
                       sycl::property_list{sycl::property::queue::enable_profiling{}});

    SYCL_GraphData sycl_data(data);

    std::vector<sycl::event> events;

    dummy_kernel(queue, sycl_data);

    auto start_glob = std::chrono::high_resolution_clock::now();
    // multi_events_BFS(queue, sycl_data, events);
    multi_frontier_BFS(queue, sycl_data, events);
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
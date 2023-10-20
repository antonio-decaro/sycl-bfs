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

void frontier_BFS(s::queue& queue, SYCL_VectorizedGraphData& data, std::vector<s::event>& events) {

    s::range<1> global{DEFAULT_WORK_GROUP_SIZE * (data.data.size())}; // each workgroup will process a graph
    s::range<1> local{DEFAULT_WORK_GROUP_SIZE};

    auto e = queue.submit([&](s::handler& cgh) {
        constexpr size_t ACC_SIZE = 8;

        size_t n_nodes [ACC_SIZE];
        s::accessor<size_t, 1, s::access::mode::read> offsets_acc[ACC_SIZE];
        s::accessor<nodeid_t, 1, s::access::mode::read> edges_acc[ACC_SIZE];
        s::accessor<nodeid_t, 1, s::access::mode::discard_write> parents_acc[ACC_SIZE];
        s::accessor<distance_t, 1, s::access::mode::read_write> distances_acc[ACC_SIZE];
        for (int i = 0; i < data.data.size(); i++) {
            offsets_acc[i] = data.offsets[i].get_access<s::access::mode::read>(cgh);
            edges_acc[i] = data.edges[i].get_access<s::access::mode::read>(cgh);
            parents_acc[i] = data.parents[i].get_access<s::access::mode::discard_write>(cgh);
            distances_acc[i] = data.distances[i].get_access<s::access::mode::read_write>(cgh);
            n_nodes[i] = data.data[i].num_nodes;
        }

        typedef int fsize_t;
        s::local_accessor<fsize_t, 1> frontier{s::range<1>{DEFAULT_WORK_GROUP_SIZE}, cgh};
        s::local_accessor<size_t, 1> fsize_curr{s::range<1>{1}, cgh};
        s::local_accessor<size_t, 1> fsize_prev{s::range<1>{1}, cgh};

        cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) {
            s::atomic_ref<size_t, s::memory_order::acq_rel, s::memory_scope::work_group> fsize_curr_ar{fsize_curr[0]};
            auto grp_id = item.get_group_linear_id();
            auto loc_id = item.get_local_id(0);

            auto distances = distances_acc[grp_id];
            auto parents = parents_acc[grp_id];
            auto offsets = offsets_acc[grp_id];
            auto edges = edges_acc[grp_id];
            auto size = n_nodes[grp_id];

            if (loc_id == 0) {
                distances[0] = 0;
                fsize_prev[0] = 1;
            } else if (loc_id < size) {
                distances[loc_id] = -1;
            }
            
            item.barrier(s::access::fence_space::local_space);
            while (fsize_prev[0] > 0) {
                if (loc_id < fsize_prev[0]) {
                    nodeid_t node = frontier[loc_id];
                    for (int i = offsets[node]; i < offsets[node + 1]; i++) {
                        nodeid_t neighbor = edges[i];
                        if (distances[neighbor] == -1) {
                            distances[neighbor] = distances[node] + 1;
                            parents[neighbor] = node;
                            auto pos = fsize_curr_ar.fetch_add(1);
                            frontier[pos] = neighbor;
                        }
                    }
                }
                item.barrier(s::access::fence_space::local_space);
                if (loc_id == 0) {
                    fsize_prev[0] = fsize_curr[0];
                    fsize_curr[0] = 0;
                }
                item.barrier(s::access::fence_space::local_space);
            }
        });
    });
    events.push_back(e);
    e.wait_and_throw();
}

template<size_t sg_size>
void frontier_BFS(s::queue& queue, SYCL_GraphData& data, std::vector<s::event>& events, const size_t wg_size) {

    s::range<1> global{wg_size * (data.host_data.graphs_offsets.size() - 1)}; // each workgroup will process a graph
    s::range<1> local{wg_size};

    auto e = queue.submit([&](s::handler& cgh) {
        s::accessor offsets_acc{data.edges_offsets, cgh, s::read_only};
        s::accessor edges_acc{data.edges, cgh, s::read_only};
        s::accessor distances_acc{data.distances, cgh, s::read_write, s::no_init};
        s::accessor parents_acc{data.parents, cgh, s::write_only, s::no_init};
        s::accessor graphs_offsets_acc{data.graphs_offests, cgh, s::read_only};
        s::accessor nodes_offsets_acc{data.nodes_offsets, cgh, s::read_only};
        s::accessor nodes_count_acc{data.nodes_count, cgh, s::read_only};

        s::stream os {8192, 128, cgh};

        typedef int fsize_t;
        s::local_accessor<fsize_t, 1> frontier{s::range<1>{wg_size}, cgh};
        s::local_accessor<size_t, 1> fsize_curr{s::range<1>{1}, cgh};
        s::local_accessor<size_t, 1> fsize_prev{s::range<1>{1}, cgh};

        cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
            s::atomic_ref<size_t, s::memory_order::acq_rel, s::memory_scope::work_group> fsize_curr_ar{fsize_curr[0]};
            auto grp_id = item.get_group_linear_id();
            auto loc_id = item.get_local_id(0);
            auto edge_offset = graphs_offsets_acc[grp_id];
            auto node_offset = nodes_offsets_acc[grp_id];
            auto node_count = nodes_count_acc[grp_id];
            auto local_size = item.get_local_range(0);

            // initi frontier
            frontier[loc_id] = 0;

            // init data
            for (int i = loc_id; i < node_count; i += local_size) {
                parents_acc[node_offset + i] = -1;
                distances_acc[node_offset + i] = -1;
            }
            distances_acc[node_offset] = 0;
            fsize_prev[0] = 1;
            
            item.barrier(s::access::fence_space::local_space);
            while (fsize_prev[0] > 0) {
                if (loc_id < fsize_prev[0]) {
                    nodeid_t node = frontier[loc_id];
                    for (int i = offsets_acc[node_offset + node]; i < offsets_acc[node_offset + node + 1]; i++) {
                        nodeid_t neighbor = edges_acc[i];
                        if (distances_acc[node_offset + neighbor] == -1) {
                            distances_acc[node_offset + neighbor] = distances_acc[node_offset + node] + 1;
                            parents_acc[node_offset + neighbor] = node;
                            auto pos = fsize_curr_ar.fetch_add(1);
                            frontier[pos] = neighbor;
                        }
                    }
                }
                item.barrier(s::access::fence_space::local_space);
                if (loc_id == 0) {
                    fsize_prev[0] = fsize_curr[0];
                    fsize_curr[0] = 0;
                }
                item.barrier(s::access::fence_space::local_space);
            }
        });
    });
    events.push_back(e);
    e.wait_and_throw();
}

class MultipleSimpleBFS {
public:
    MultipleSimpleBFS(std::vector<CSRHostData>& data) : 
        data(data) {}

    template<size_t sg_size, bool init = false>
    void run(const size_t wg_size = DEFAULT_WORK_GROUP_SIZE) {
            // s queue definition
        s::queue queue (s::gpu_selector_v, 
                        s::property_list{s::property::queue::enable_profiling{}});

        CompressedHostData compressed_data(data);
        SYCL_GraphData sycl_data(compressed_data);
        SYCL_VectorizedGraphData sycl_vectorized_data(data);

        std::vector<s::event> events;

        auto start_glob = std::chrono::high_resolution_clock::now();
        frontier_BFS<sg_size>(queue, sycl_data, events, wg_size);
        auto end_glob = std::chrono::high_resolution_clock::now();

        long duration = 0;
        for (s::event& e : events) {
            auto start = e.get_profiling_info<s::info::event_profiling::command_start>();
            auto end = e.get_profiling_info<s::info::event_profiling::command_end>();
            duration += (end - start);
        }

        sycl_data.write_back();
        
        if (!init) {
            std::cout << "Sub-grup size: " << sg_size << std::endl;
            std::cout << "- Kernels duration: " << duration / 1000 << " us" << std::endl;
            std::cout << "- Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count() << " us" << std::endl;
        }
    }

private:
    std::vector<CSRHostData>& data;
};

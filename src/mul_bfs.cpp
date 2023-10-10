#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "kernel_sizes.hpp"
#include "bfs.hpp"

// TODO fix: only the first graph is processed correctly

namespace s = sycl;

class CompressedHostData {
public:
    CompressedHostData(std::vector<CSRHostData>& data) : data(data) {
        int total_nodes = 0;
        for (int i = 0; i < data.size(); i++) {
            auto d = this->data[i];
            if (i != 0) {
                d.csr.offsets.erase(d.csr.offsets.begin());
            }

            for (int j = 0; j < d.csr.offsets.size(); j++) {
                compressed_offsets.push_back(total_offset_size + d.csr.offsets[j]);
            }
            compressed_edges.insert(compressed_edges.end(), d.csr.edges.begin(), d.csr.edges.end());
            
            nodes_count.push_back(d.num_nodes);
            nodes_offsets.push_back(total_nodes);
            graphs_offsets.push_back(total_offset_size);
            
            total_offset_size += d.csr.offsets.back();
            total_nodes += d.num_nodes;
        }
        graphs_offsets.push_back(total_offset_size);
        nodes_offsets.push_back(total_nodes);

        compressed_distances = std::vector<distance_t>(total_nodes, -1);
        compressed_parents = std::vector<nodeid_t>(total_nodes, -1);
    }

    void write_back() {
        size_t k = 0;
        for (auto& d : data) {
            for (size_t i = 0; i < d.num_nodes; i++) {
                d.distances[i] = compressed_distances[k];
                d.parents[i] = compressed_parents[k];
                k++;
            }
        }
    }

    std::vector<CSRHostData>& data;
    size_t total_offset_size = 0;
    std::vector<size_t> compressed_offsets, nodes_count, graphs_offsets, nodes_offsets;
    std::vector<distance_t> compressed_distances;
    std::vector<nodeid_t> compressed_edges, compressed_parents;
};

class SYCL_GraphData {
public:
    SYCL_GraphData(CompressedHostData& data) :
        host_data(data),
        nodes_offsets(s::buffer<size_t, 1>(data.nodes_offsets.data(), s::range{data.nodes_offsets.size()})),
        graphs_offests(s::buffer<size_t, 1>(data.graphs_offsets.data(), s::range{data.graphs_offsets.size()})),
        nodes_count(s::buffer<size_t, 1>(data.nodes_count.data(), s::range{data.nodes_count.size()})),
        edges_offsets(s::buffer<size_t, 1>{data.compressed_offsets.data(), s::range{data.compressed_offsets.size()}}),
        edges(s::buffer<nodeid_t, 1>{data.compressed_edges.data(), s::range{data.compressed_edges.size()}}),
        distances(s::buffer<distance_t, 1>{data.compressed_distances.data(), s::range{data.compressed_distances.size()}}),
        parents(s::buffer<nodeid_t, 1>{data.compressed_parents.data(), s::range{data.compressed_parents.size()}}) {}

    void write_back() {
        auto dacc = distances.get_host_access();
        auto pacc = parents.get_host_access();
        for (int i = 0; i < host_data.compressed_distances.size(); i++) {
            host_data.compressed_distances[i] = dacc[i];
            host_data.compressed_parents[i] = pacc[i];
        }
        
        host_data.write_back();
    }

    CompressedHostData& host_data;
    s::buffer<nodeid_t, 1> edges, parents;
    s::buffer<distance_t, 1> distances;
    s::buffer<size_t, 1> graphs_offests, nodes_offsets, nodes_count, edges_offsets;
};

void multi_frontier_BFS(s::queue& queue, SYCL_GraphData& data, std::vector<s::event>& events) {

    s::range<1> global{WORK_GROUP_SIZE * (data.host_data.graphs_offsets.size() - 1)}; // each workgroup will process a graph
    s::range<1> local{WORK_GROUP_SIZE};

    auto e = queue.submit([&](s::handler& cgh) {
        auto offsets_acc = data.edges_offsets.get_access<s::access::mode::read>(cgh);
        auto edges_acc = data.edges.get_access<s::access::mode::read>(cgh);
        auto distances_acc = data.distances.get_access<s::access::mode::discard_read_write>(cgh);
        auto parents_acc = data.parents.get_access<s::access::mode::discard_write>(cgh);
        auto graphs_offsets_acc = data.graphs_offests.get_access<s::access::mode::read>(cgh);
        auto nodes_offsets_acc = data.nodes_offsets.get_access<s::access::mode::read>(cgh);
        auto nodes_count_acc = data.nodes_count.get_access<s::access::mode::read>(cgh);

        s::stream os {8192, 128, cgh};

        typedef int fsize_t;
        s::local_accessor<fsize_t, 1> frontier{s::range<1>{WORK_GROUP_SIZE}, cgh};
        s::local_accessor<size_t, 1> fsize_curr{s::range<1>{1}, cgh};
        s::local_accessor<size_t, 1> fsize_prev{s::range<1>{1}, cgh};

        cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) {
            s::atomic_ref<size_t, s::memory_order::acq_rel, s::memory_scope::work_group> fsize_curr_ar{fsize_curr[0]};
            auto grid = item.get_group_linear_id();
            auto lid = item.get_local_id(0);
            auto edge_offset = graphs_offsets_acc[grid];
            auto node_offset = nodes_offsets_acc[grid];
            auto n_nodes = graphs_offsets_acc[grid] - edge_offset;

            if (lid == 0) {
                os << "Edge Offset: " << edge_offset << " | Node Offset: " << node_offset << " | First node: " << offsets_acc[edge_offset] << "\n";
                distances_acc[edge_offset] = 0;
                fsize_prev[0] = 1;
            } else if (lid < nodes_count_acc[item.get_group(0)]) {
                distances_acc[edge_offset + lid] = -1;
            }
            
            item.barrier(s::access::fence_space::local_space);
            while (fsize_prev[0] > 0) {
                if (lid < fsize_prev[0]) {
                    nodeid_t node = frontier[lid];
                    for (int i = offsets_acc[edge_offset + node]; i < offsets_acc[edge_offset + node + 1]; i++) {
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
                if (lid == 0) {
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

void MultipleSimpleBFS::run() {

    // s queue definition
    s::queue queue (s::gpu_selector_v, 
                       s::property_list{s::property::queue::enable_profiling{}});

    CompressedHostData compressed_data(data);
    SYCL_GraphData sycl_data(compressed_data);

    std::vector<s::event> events;

    auto start_glob = std::chrono::high_resolution_clock::now();
    multi_frontier_BFS(queue, sycl_data, events);
    auto end_glob = std::chrono::high_resolution_clock::now();

    long duration = 0;
    for (s::event& e : events) {
        auto start = e.get_profiling_info<s::info::event_profiling::command_start>();
        auto end = e.get_profiling_info<s::info::event_profiling::command_end>();
        duration += (end - start);
    }

    sycl_data.write_back();

    std::cout << "[*] Kernels duration: " << duration / 1000 << " us" << std::endl;
    std::cout << "[*] Total duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count() << " us" << std::endl;
}
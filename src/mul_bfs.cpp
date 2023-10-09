#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "kernel_sizes.hpp"
#include "bfs.hpp"


namespace s = sycl;

class CompressedHostData {
public:
    CompressedHostData(std::vector<CSRHostData&> data) : data(data) {
        for (int i = 0; i < data.size(); i++) {
            auto d = this->data[i];
            if (i != 0) {
                d.csr.offsets.erase(d.csr.offsets.begin());
            }

            for (int j = 0; i < d.csr.offsets.size() - 1; i++) {
                compressed_offsets.push_back(total_size + d.csr.offsets[j]);
            }
            compressed_edges.insert(compressed_edges.end(), std::make_move_iterator(d.csr.edges.begin()), std::make_move_iterator(d.csr.edges.end()));
            compressed_distances.insert(compressed_distances.end(), std::make_move_iterator(d.distances.begin()), std::make_move_iterator(d.distances.end()));
            compressed_parents.insert(compressed_parents.end(), std::make_move_iterator(d.parents.begin()), std::make_move_iterator(d.parents.end()));
            nodes_count.push_back(d.num_nodes);
            graphs_offsets.push_back(total_size);
            
            total_size += d.csr.offsets.back();
        }
        graphs_offsets.push_back(total_size);
    }

    void write_back() {
        size_t off = 0;
        for (size_t i = 0; i < data.size(); i++) {
            data[i].distances = std::vector<nodeid_t>(compressed_distances.begin() + off, compressed_distances.begin() + graphs_offsets[i]);
            data[i].parents = std::vector<nodeid_t>(compressed_parents.begin() + off, compressed_parents.begin() + graphs_offsets[i]);
            off = graphs_offsets[i];
        }
    }

    std::vector<CSRHostData&> data;
    size_t total_size = 0;
    std::vector<size_t> compressed_offsets, nodes_count, graphs_offsets;
    std::vector<nodeid_t> compressed_edges, compressed_distances, compressed_parents;
};

class SYCL_GraphData {
public:
    SYCL_GraphData(CompressedHostData& data) :
        host_data(data),
        graphs_offests(s::buffer<size_t, 1>(data.graphs_offsets.data(), s::range{data.graphs_offsets.size()})),
        nodes_count(s::buffer<size_t, 1>(data.nodes_count.data(), s::range{data.nodes_count.size()})),
        offsets(s::buffer<size_t, 1>{data.compressed_offsets.data(), s::range{data.compressed_offsets.size()}}),
        edges(s::buffer<nodeid_t, 1>{data.compressed_edges.data(), s::range{data.compressed_edges.size()}}),
        distances(s::buffer<nodeid_t, 1>{data.compressed_distances.data(), s::range{data.compressed_distances.size()}}),
        parents(s::buffer<nodeid_t, 1>{data.compressed_parents.data(), s::range{data.compressed_parents.size()}}) {}

    void write_back() {
        distances.set_final_data(host_data.compressed_distances.data());
        parents.set_final_data(host_data.compressed_parents.data());
        distances.set_write_back(true);
        parents.set_write_back(true);
        host_data.write_back();
    }

    CompressedHostData& host_data;
    s::buffer<nodeid_t, 1> edges, distances, parents;
    s::buffer<size_t, 1> graphs_offests, nodes_count, offsets;
};

void dummy_kernel(s::queue& queue, SYCL_GraphData& data) {

    // dummy kernel to init data
    queue.submit([&](s::handler& h) {
        auto offsets_acc = data.offsets.get_access<s::access::mode::read>(h);
        auto edges_acc = data.edges.get_access<s::access::mode::read>(h);
        auto distances_acc = data.distances.get_access<s::access::mode::discard_write>(h);
        auto parents_acc = data.parents.get_access<s::access::mode::discard_write>(h);

        h.parallel_for(s::range<1>{data.host_data.graphs_offsets.back()}, [=](s::id<1> idx) {
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

void multi_frontier_BFS(s::queue& queue, SYCL_GraphData& data, std::vector<s::event>& events) {

    s::range<1> global{WORK_GROUP_SIZE * data.host_data.graphs_offsets.size()}; // each workgroup will process a graph
    s::range<1> local{WORK_GROUP_SIZE}; 

    auto e = queue.submit([&](s::handler& cgh) {
        auto offsets_acc = data.offsets.get_access<s::access::mode::read>(cgh);
        auto edges_acc = data.edges.get_access<s::access::mode::read>(cgh);
        auto distances_acc = data.distances.get_access<s::access::mode::read_write>(cgh);
        auto parents_acc = data.parents.get_access<s::access::mode::discard_write>(cgh);
        auto graphs_offsets_acc = data.graphs_offests.get_access<s::access::mode::read>(cgh);
        auto nodes_count_acc = data.nodes_count.get_access<s::access::mode::read>(cgh);

        s::local_accessor<nodeid_t, 1> frontier(s::range<1>(WORK_GROUP_SIZE), cgh);
        s::local_accessor<nodeid_t, 1> f_size(s::range<1>(1), cgh);
        s::local_accessor<nodeid_t, 1> fold_size(s::range<1>(1), cgh);

        cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> idx) {
            s::atomic_ref<nodeid_t, s::memory_order::acq_rel, s::memory_scope::work_group> f_size_ar{f_size[0]};
            s::atomic_ref<nodeid_t, s::memory_order::acq_rel, s::memory_scope::work_group> fold_size_ar{fold_size[0]};
            auto grp = idx.get_group();
            auto grp_id = idx.get_group_linear_id();
            auto offset = graphs_offsets_acc[grp_id];
            auto num_nodes = nodes_count_acc[grp_id];
            auto lid = idx.get_local_id(0);

            // init frontier mask
            if (idx.get_local_linear_id() == 0) {
                fold_size_ar.store(1);
                f_size_ar.store(0);
            }
            idx.barrier(s::access::fence_space::local_space);

            while (fold_size_ar.load()) {
                // process frontier
                // extract the node id from the mask
                if (lid < f_size_ar.load()) {
                    int node = frontier[lid];
                    for (int i = offsets_acc[offset + node]; i < offsets_acc[offset + node + 1]; i++) {
                        int neighbor = offset + edges_acc[i];
                        if (distances_acc[offset + neighbor] == -1) {
                            distances_acc[neighbor] = distances_acc[node] + 1;
                            parents_acc[neighbor] = node;
                            auto pos = f_size_ar.fetch_add(1);
                            frontier[pos] = neighbor;
                        }
                    }
                }
                idx.barrier(s::access::fence_space::local_space);
                if (idx.get_local_id(0) == 0) {
                    fold_size_ar.store(f_size_ar.load());
                    f_size_ar.store(0);
                }
                idx.barrier(s::access::fence_space::local_space);
            }
        });
    });

    events.push_back(e);
    e.wait();
}

void MultipleSimpleBFS::run() {

    // s queue definition
    s::queue queue (s::gpu_selector_v, 
                       s::property_list{s::property::queue::enable_profiling{}});


    CompressedHostData compressed_data(data);
    SYCL_GraphData sycl_data(compressed_data);

    std::vector<s::event> events;

    dummy_kernel(queue, sycl_data);

    auto start_glob = std::chrono::high_resolution_clock::now();
    // multi_events_BFS(queue, sycl_data, events);
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
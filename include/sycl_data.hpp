#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include "host_data.hpp"
#include "types.hpp"

#ifndef SYCL_DATA_HPP
#define SYCL_DATA_HPP

class SYCL_VectorizedGraphData {
public:
    SYCL_VectorizedGraphData(std::vector<CSRHostData>& data) : data(data) {

        for (auto& d : data) {
            offsets.push_back(sycl::buffer<size_t, 1>{d.csr.offsets.data(), sycl::range{d.csr.offsets.size()}});
            edges.push_back(sycl::buffer<nodeid_t, 1>{d.csr.edges.data(), sycl::range{d.csr.edges.size()}});
            parents.push_back(sycl::buffer<nodeid_t, 1>{d.parents.data(), sycl::range{d.parents.size()}});
            distances.push_back(sycl::buffer<distance_t, 1>{d.distances.data(), sycl::range{d.distances.size()}});
        }
    }

    void write_back() {
        for (int i = 0; i < data.size(); i++) {
            auto& d = data[i];
            parents[i].set_final_data(d.parents.data());
            parents[i].set_write_back(true);
            distances[i].set_final_data(d.distances.data());
            distances[i].set_write_back(true);
        }
    }

    std::vector<CSRHostData>& data;
    std::vector<sycl::buffer<size_t, 1>> offsets;
    std::vector<sycl::buffer<nodeid_t, 1>> edges;
    std::vector<sycl::buffer<nodeid_t, 1>> parents;
    std::vector<sycl::buffer<distance_t, 1>> distances;
};

class SYCL_CompressedGraphData {
public:
    SYCL_CompressedGraphData(CompressedHostData& data) :
        host_data(data),
        nodes_offsets(sycl::buffer<size_t, 1>(data.nodes_offsets.data(), sycl::range{data.nodes_offsets.size()})),
        graphs_offests(sycl::buffer<size_t, 1>(data.graphs_offsets.data(), sycl::range{data.graphs_offsets.size()})),
        nodes_count(sycl::buffer<size_t, 1>(data.nodes_count.data(), sycl::range{data.nodes_count.size()})),
        edges_offsets(sycl::buffer<size_t, 1>{data.compressed_offsets.data(), sycl::range{data.compressed_offsets.size()}}),
        edges(sycl::buffer<nodeid_t, 1>{data.compressed_edges.data(), sycl::range{data.compressed_edges.size()}}),
        distances(sycl::buffer<distance_t, 1>{data.compressed_distances.data(), sycl::range{data.compressed_distances.size()}}),
        parents(sycl::buffer<nodeid_t, 1>{data.compressed_parents.data(), sycl::range{data.compressed_parents.size()}}) {}

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
    sycl::buffer<nodeid_t, 1> edges, parents;
    sycl::buffer<distance_t, 1> distances;
    sycl::buffer<size_t, 1> graphs_offests, nodes_offsets, nodes_count, edges_offsets;
};

#endif
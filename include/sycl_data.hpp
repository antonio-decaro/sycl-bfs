#include <sycl/sycl.hpp>
#include <vector>
#include "host_data.hpp"
#include "types.hpp"

#ifndef SYCL_CSR_GRAPH_HPP
#define SYCL_CSR_GRAPH_HPP

class SYCL_GraphData {
public:
    SYCL_GraphData(CSRHostData& data) :
        host_data(data),
        num_nodes(data.num_nodes),
        offsets(sycl::buffer<nodeid_t, 1>{data.csr.offsets.data(), sycl::range{data.csr.offsets.size()}}),
        edges(sycl::buffer<nodeid_t, 1>{data.csr.edges.data(), sycl::range{data.csr.edges.size()}}),
        distances(sycl::buffer<nodeid_t, 1>{data.distances.data(), sycl::range{data.distances.size()}}),
        parents(sycl::buffer<nodeid_t, 1>{data.parents.data(), sycl::range{data.parents.size()}})
    {
        distances.set_write_back(false);
        parents.set_write_back(false);
    }

    size_t num_nodes;
    CSRHostData& host_data;
    sycl::buffer<nodeid_t, 1> offsets, edges, distances, parents;
};

#endif
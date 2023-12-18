#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include "graph.hpp"
#include "types.hpp"

namespace sygraph {
namespace details {

class SYGraph {
private:
	sycl::buffer<size_t, 1> row_offsets_buf;
	sycl::buffer<node_t, 1> col_indices_buf;
	sycl::buffer<label_t, 1> node_labels_buf;
public:
	SYGraph() = delete;

	SYGraph(const std::vector<size_t>& row_offsets, const std::vector<node_t>& col_indices) :
		row_offsets_buf{sycl::buffer<size_t, 1>{row_offsets.data(), sycl::range{row_offsets.size()}}},
		col_indices_buf{sycl::buffer<node_t, 1>{col_indices.data(), sycl::range{col_indices.size()}}},
		node_labels_buf{sycl::buffer<label_t, 1>{0}}
	{}

	SYGraph(const std::vector<size_t>& row_offsets, const std::vector<node_t>& col_indices, const std::vector<label_t>& node_labels) :
		// Graph(row_offsets, col_indices, node_labels), 
		row_offsets_buf(sycl::buffer<size_t, 1>{row_offsets.data(), sycl::range{row_offsets.size()}}),
		col_indices_buf(sycl::buffer<node_t, 1>{col_indices.data(), sycl::range{col_indices.size()}}),
		node_labels_buf(sycl::buffer<node_t, 1>{node_labels.data(), sycl::range{node_labels.size()}})
	{}

	const size_t getSize() const { return row_offsets_buf.size() - 1; }

	inline sycl::accessor<size_t, 1, sycl::access::mode::read> getRowOffsetsDeviceAccessor(sycl::handler& h) { 
		return sycl::accessor {this->row_offsets_buf, h}; 
	}

	inline sycl::accessor<node_t, 1, sycl::access::mode::read> getColIndicesDeviceAccessor(sycl::handler& h) { 
		return sycl::accessor {this->col_indices_buf, h}; 
	}

	inline sycl::accessor<label_t, 1, sycl::access::mode::read> getNodeLabelsDeviceAccessor(sycl::handler& h) { 
		return sycl::accessor {this->node_labels_buf, h}; 
	}

	inline sycl::host_accessor<size_t, 1> getRowOffsetsHostAccessor() { 
		return sycl::host_accessor {this->row_offsets_buf}; 
	}

	inline sycl::host_accessor<node_t, 1> getColIndicesHostAccessor() { 
		return sycl::host_accessor {this->col_indices_buf}; 
	}

	inline sycl::host_accessor<label_t, 1> getNodeLabelsHostAccessor() { 
		return sycl::host_accessor {this->node_labels_buf}; 
	}
};


} // namespace details
} // namespace sygraph

// class SYCL_CompressedGraphData
// {
// public:
// 	SYCL_CompressedGraphData(CompressedHostData &data) : 
// 		host_data(data),
// 		nodes_offsets(sycl::buffer<size_t, 1>(data.nodes_offsets.data(), sycl::range{data.nodes_offsets.size()})),
// 		graphs_offests(sycl::buffer<size_t, 1>(data.graphs_offsets.data(), sycl::range{data.graphs_offsets.size()})),
// 		nodes_count(sycl::buffer<size_t, 1>(data.nodes_count.data(), sycl::range{data.nodes_count.size()})),
// 		edges_offsets(sycl::buffer<size_t, 1>{data.compressed_offsets.data(), sycl::range{data.compressed_offsets.size()}}),
// 		edges(sycl::buffer<nodeid_t, 1>{data.compressed_edges.data(), sycl::range{data.compressed_edges.size()}}),
// 		parents(sycl::buffer<nodeid_t, 1>{data.compressed_parents.data(), sycl::range{data.compressed_parents.size()}}) {}

// 	sycl::event init(sycl::queue &q, const std::vector<nodeid_t> &sources, size_t wg_size = DEFAULT_WORK_GROUP_SIZE)
// 	{
// 		sycl::buffer<nodeid_t, 1> device_source{sources.data(), sycl::range{sources.size()}};

// 		return q.submit([&](sycl::handler &h) {
// 			sycl::range global {host_data.num_graphs * wg_size};
// 			sycl::range local {wg_size};

// 			sycl::accessor sources {device_source, h, sycl::read_only};
// 			sycl::accessor nodes_acc{nodes_offsets, h, sycl::read_only};
// 			sycl::accessor nodes_count_acc{nodes_count, h, sycl::read_only};
// 			sycl::accessor parents_acc{parents, h, sycl::write_only, sycl::no_init};

// 			h.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
// 				auto gid = item.get_group_linear_id();
// 				auto lid = item.get_local_linear_id();
// 				auto local_range = item.get_local_range(0);
// 				auto nodes_count = nodes_count_acc[gid];
// 				auto source = sources[gid];

// 				for (int i = lid; i < nodes_count; i += local_range) {
// 					parents_acc[nodes_acc[gid] + i] = -1;
// 					if (i == source) {
// 						parents_acc[nodes_acc[gid] + i] = source;
// 					}
// 				}
// 			}); 
// 		});
// 	}

// 	void write_back()
// 	{
// 		auto pacc = parents.get_host_access();
// 		for (int i = 0; i < host_data.compressed_parents.size(); i++)
// 		{
// 			host_data.compressed_parents[i] = pacc[i];
// 		}

// 		host_data.write_back();
// 	}

// 	CompressedHostData &host_data;
// 	sycl::buffer<nodeid_t, 1> edges, parents;
// 	sycl::buffer<size_t, 1> graphs_offests, nodes_offsets, nodes_count, edges_offsets;
// };

// class SYCL_SimpleGraphData
// {
// public:
// 	SYCL_SimpleGraphData(CSRHostData &data) : 
// 		host_data(data),
// 		num_nodes(data.num_nodes),
// 		edges_offsets(sycl::buffer<size_t, 1>{data.csr.offsets.data(), sycl::range{data.csr.offsets.size()}}),
// 		edges(sycl::buffer<nodeid_t, 1>{data.csr.edges.data(), sycl::range{data.csr.edges.size()}}),
// 		parents(sycl::buffer<nodeid_t, 1>{data.parents.data(), sycl::range{data.parents.size()}})
// 	{
// 		parents.set_write_back(false);
// 	}

// 	void write_back()
// 	{
// 		parents.set_final_data(host_data.parents.data());
// 		parents.set_write_back(true);
// 	}

// 	sycl::event init(sycl::queue &q, const nodeid_t source) {
// 		return q.submit([&](sycl::handler &h) {
// 			sycl::accessor parents_acc{parents, h, sycl::write_only, sycl::no_init};

// 			h.parallel_for(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
// 				int node = idx[0];
// 				parents_acc[node] = -1;
// 				if (node == source) {
// 					parents_acc[node] = node;
// 				}
// 			});
// 		});
// 	}

// 	size_t num_nodes;
// 	CSRHostData &host_data;
// 	sycl::buffer<nodeid_t, 1> parents, edges;
// 	sycl::buffer<size_t, 1> edges_offsets;
// };

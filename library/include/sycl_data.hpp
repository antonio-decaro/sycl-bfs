#pragma once

#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include "graph.hpp"
#include "compressed.hpp"
#include "types.hpp"

namespace sygraph {

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

class SYGraphsCompressed {
private:
	sycl::buffer<node_t, 1> col_indices;
	sycl::buffer<size_t, 1> graphs_offests, nodes_offsets, nodes_count, edges_offsets;
	size_t num_graphs;
public:
	SYGraphsCompressed(details::CompressedGraphs& data) : 
		num_graphs(data.getNumGraphs()),
		nodes_offsets(sycl::buffer<size_t, 1>(data.getNodesOffsets().data(), sycl::range{data.getNodesOffsets().size()})),
		graphs_offests(sycl::buffer<size_t, 1>(data.getGraphsOffsets().data(), sycl::range{data.getGraphsOffsets().size()})),
		nodes_count(sycl::buffer<size_t, 1>(data.getNodesCount().data(), sycl::range{data.getNodesCount().size()})),
		edges_offsets(sycl::buffer<size_t, 1>{data.getCompressedOffsets().data(), sycl::range{data.getCompressedOffsets().size()}}),
		col_indices(sycl::buffer<node_t, 1>{data.getCompressedColIndices().data(), sycl::range{data.getCompressedColIndices().size()}}) {}

	inline sycl::accessor<size_t, 1, sycl::access::mode::read> getNodesOffsetsDeviceAccessor(sycl::handler& h) {
		return sycl::accessor {this->nodes_offsets, h};
	}

	inline sycl::accessor<size_t, 1, sycl::access::mode::read> getGraphsOffsetsDeviceAccessor(sycl::handler& h) {
		return sycl::accessor {this->graphs_offests, h};
	}

	inline sycl::accessor<size_t, 1, sycl::access::mode::read> getNodesCountDeviceAccessor(sycl::handler& h) {
		return sycl::accessor {this->nodes_count, h};
	}

	inline sycl::accessor<size_t, 1, sycl::access::mode::read> getEdgesOffsetsDeviceAccessor(sycl::handler& h) {
		return sycl::accessor {this->edges_offsets, h};
	}

	inline sycl::accessor<node_t, 1, sycl::access::mode::read> getColIndicesDeviceAccessor(sycl::handler& h) {
		return sycl::accessor {this->col_indices, h};
	}

	inline sycl::host_accessor<size_t, 1> getNodesOffsetsHostAccessor() {
		return sycl::host_accessor {this->nodes_offsets};
	}

	inline sycl::host_accessor<size_t, 1> getGraphsOffsetsHostAccessor() {
		return sycl::host_accessor {this->graphs_offests};
	}

	inline sycl::host_accessor<size_t, 1> getNodesCountHostAccessor() {
		return sycl::host_accessor {this->nodes_count};
	}

	inline sycl::host_accessor<size_t, 1> getEdgesOffsetsHostAccessor() {
		return sycl::host_accessor {this->edges_offsets};
	}

	inline sycl::host_accessor<node_t, 1> getColIndicesHostAccessor() {
		return sycl::host_accessor {this->col_indices};
	}

	inline size_t getNumGraphs() const { return this->num_graphs; }
};

} // namespace sygraph
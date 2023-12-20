#pragma once

#include "types.hpp"
#include "graph.hpp"
#include <vector>

namespace sygraph {
namespace details {

class CompressedGraphs {
private:
  size_t num_graphs;
	std::vector<Graph> &data;
	size_t total_offset_size = 0;
	std::vector<size_t> compressed_offsets, nodes_count, graphs_offsets, nodes_offsets;
	std::vector<node_t> compressed_col_indices;

public:
  CompressedGraphs(std::vector<Graph> &data) : data(data)
	{
		num_graphs = data.size();
		int total_nodes = 0;
		for (int i = 0; i < data.size(); i++)
		{
			std::vector<size_t> new_offsets { data[i].getRowOffsets() };
			if (i != 0)
			{
				new_offsets.erase(new_offsets.begin());
			}

			for (int j = 0; j < new_offsets.size(); j++)
			{
				compressed_offsets.push_back(total_offset_size + new_offsets[j]);
			}
			compressed_col_indices.insert(compressed_col_indices.end(), data[i].getColIndices().begin(), data[i].getColIndices().end());

			nodes_count.push_back(data[i].getSize());
			nodes_offsets.push_back(total_nodes);
			graphs_offsets.push_back(total_offset_size);

			total_offset_size += data[i].getRowOffsets().back();
			total_nodes += data[i].getSize();
		}
		graphs_offsets.push_back(total_offset_size);
		nodes_offsets.push_back(total_nodes);
	}

  size_t getNumGraphs() const { return num_graphs; }
  size_t getTotalOffsetSize() const { return total_offset_size; }
  const std::vector<size_t> &getCompressedOffsets() const { return compressed_offsets; }
  const std::vector<size_t> &getNodesCount() const { return nodes_count; }
  const std::vector<size_t> &getGraphsOffsets() const { return graphs_offsets; }
  const std::vector<size_t> &getNodesOffsets() const { return nodes_offsets; }
  const std::vector<node_t> &getCompressedColIndices() const { return compressed_col_indices; }
};

} // namespace details
} // namespace sygraph

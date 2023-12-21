#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <cstddef>
#include <vector>
#include <filesystem>
#include "types.hpp"
#include "host_data.hpp"

CSRHostData readGraphFromFile(std::string filename, bool labels = false) {

	size_t num_nodes;
  size_t num_edges;

	std::ifstream file(filename);
	file >> num_nodes;
  file >> num_edges;

	std::vector<size_t> row_offsets(num_nodes + 1, 0);
	std::vector<nodeid_t> parents(num_nodes, 0);
	std::vector<nodeid_t> col_indices(num_edges, 0);
	std::vector<int> node_labels(num_edges, 0);

	if (labels) {
		for (int i = 0; i < num_nodes; i++)
		{
			file >> node_labels[i];
		}
	}

	int src, dst;
	for (int i = 0; i < num_edges; i++)
	{
		file >> src >> dst;
		row_offsets[src + 1]++;
		col_indices[i] = dst;
	}
	file.close();

	for (int i = 1; i < row_offsets.size(); i++) {
		row_offsets[i] += row_offsets[i - 1];
	}

	CSRHostData ret;
	ret.csr.offsets = row_offsets;
	ret.csr.edges = col_indices;
	ret.num_nodes = num_nodes;
	ret.parents = parents;

	return ret;
}

#endif
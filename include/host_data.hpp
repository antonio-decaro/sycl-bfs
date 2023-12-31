#include <vector>
#include "types.hpp"

#ifndef __CSR_GRAPH_HPP__
#define __CSR_GRAPH_HPP__

typedef struct
{
	std::vector<size_t> offsets;
	std::vector<nodeid_t> edges;
} CSR;

typedef struct
{
	size_t num_nodes;
	CSR csr;
	std::vector<nodeid_t> parents;
} CSRHostData;

typedef struct
{
	size_t num_nodes;
	std::vector<std::vector<adjidx_t>> adj_matrix;
} MatrixHostData;

class CompressedHostData
{
public:
	CompressedHostData(std::vector<CSRHostData> &data) : data(data)
	{
		num_graphs = data.size();
		int total_nodes = 0;
		for (int i = 0; i < data.size(); i++)
		{
			std::vector<size_t> new_offsets { data[i].csr.offsets };
			if (i != 0)
			{
				new_offsets.erase(new_offsets.begin());
			}

			for (int j = 0; j < new_offsets.size(); j++)
			{
				compressed_offsets.push_back(total_offset_size + new_offsets[j]);
			}
			compressed_edges.insert(compressed_edges.end(), data[i].csr.edges.begin(), data[i].csr.edges.end());

			nodes_count.push_back(data[i].num_nodes);
			nodes_offsets.push_back(total_nodes);
			graphs_offsets.push_back(total_offset_size);

			total_offset_size += data[i].csr.offsets.back();
			total_nodes += data[i].num_nodes;
		}
		graphs_offsets.push_back(total_offset_size);
		nodes_offsets.push_back(total_nodes);

		compressed_parents = std::vector<nodeid_t>(total_nodes, -1);
	}

	void write_back()
	{
		size_t k = 0;
		for (auto &d : data)
		{
			for (size_t i = 0; i < d.num_nodes; i++)
			{
				d.parents[i] = compressed_parents[k];
				k++;
			}
		}
	}

	size_t num_graphs;
	std::vector<CSRHostData> &data;
	size_t total_offset_size = 0;
	std::vector<size_t> compressed_offsets, nodes_count, graphs_offsets, nodes_offsets;
	std::vector<nodeid_t> compressed_edges, compressed_parents;
};

#endif

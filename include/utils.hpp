#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <fstream>
#include <cstddef>
#include <vector>
#include <filesystem>
#include "types.hpp"
#include "host_data.hpp"

CSR build_csr(const std::vector<std::vector<nodeid_t>> &adjacency_list)
{
	CSR csr;
	size_t num_nodes = adjacency_list.size();
	csr.offsets.resize(num_nodes + 1);
	nodeid_t off = 0;
	for (nodeid_t i = 0; i < num_nodes; i++)
	{
		csr.offsets[i] = off;
		off += adjacency_list[i].size();
	}
	csr.offsets[num_nodes] = off;

	csr.edges.resize(off);
	for (nodeid_t i = 0; i < num_nodes; i++)
	{
		for (nodeid_t j = 0; j < adjacency_list[i].size(); j++)
		{
			csr.edges[csr.offsets[i] + j] = adjacency_list[i][j];
		}
	}
	return csr;
}

std::vector<std::vector<adjidx_t>> build_adj_matrix(const std::vector<std::vector<nodeid_t>> &adjacency_list)
{
	size_t num_nodes = adjacency_list.size();
	std::vector<std::vector<adjidx_t>> adj_matrix(num_nodes);
	for (nodeid_t i = 0; i < num_nodes; i++)
	{
		adj_matrix[i].resize(num_nodes);
		for (nodeid_t j = 0; j < num_nodes; j++)
		{
			adj_matrix[i][j] = 0;
		}
	}
	for (nodeid_t i = 0; i < num_nodes; i++)
	{
		for (nodeid_t j = 0; j < adjacency_list[i].size(); j++)
		{
			adj_matrix[i][adjacency_list[i][j]] = 1;
		}
	}
	return adj_matrix;
}

CSRHostData build_csr_host_data(const std::vector<std::vector<nodeid_t>> &adjacency_list)
{
	CSRHostData host_data;
	host_data.csr = build_csr(adjacency_list);
	host_data.num_nodes = host_data.csr.offsets.size() - 1;
	host_data.distances.resize(host_data.num_nodes);
	std::fill(host_data.distances.begin(), host_data.distances.end(), -1);
	host_data.parents.resize(host_data.num_nodes);
	std::fill(host_data.parents.begin(), host_data.parents.end(), -1);
	return host_data;
}

MatrixHostData build_adj_matrix(std::string path)
{
	MatrixHostData host_data;
	std::ifstream file(path);
	file >> host_data.num_nodes;
	host_data.adj_matrix = std::vector<std::vector<adjidx_t>>(
			host_data.num_nodes,
			std::vector<adjidx_t>(host_data.num_nodes, 0));
	int src, dst;
	while (file >> src >> dst)
	{
		host_data.adj_matrix[src][dst] = 1;
	}
	file.close();

	return host_data;
}

std::vector<std::vector<nodeid_t>> read_graph_from_file(std::string path, bool undirected = false)
{
	std::vector<std::vector<nodeid_t>> adjacency_list;
	size_t num_nodes;

	std::ifstream file(path);
	int num_edges = 0;
	file >> num_nodes;
	adjacency_list.resize(num_nodes);
	int src, dst;
	while (file >> src >> dst)
	{
		adjacency_list[src].push_back(dst);
		if (undirected)
			adjacency_list[dst].push_back(src);
		num_edges++;
	}
	file.close();

	return adjacency_list;
}

#endif
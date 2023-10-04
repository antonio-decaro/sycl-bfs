#include <fstream>
#include <cstddef>
#include <vector>
#include "types.hpp"
#include "host_data.hpp"

CSR build_csr(const std::vector<std::vector<index_type>>& adjacency_list) {
    CSR csr;
    size_t num_nodes = adjacency_list.size();
    csr.offsets.resize(num_nodes + 1);
    index_type off = 0;
    for (index_type i = 0; i < num_nodes; i++) {
        csr.offsets[i] = off;
        off += adjacency_list[i].size();
    }
    csr.offsets[num_nodes] = off;

    csr.edges.resize(off);
    for (index_type i = 0; i < num_nodes; i++) {
        for (index_type j = 0; j < adjacency_list[i].size(); j++) {
            csr.edges[csr.offsets[i] + j] = adjacency_list[i][j];
        }
    }
    return csr;
}

HostData build_host_data(const std::vector<std::vector<index_type>>& adjacency_list) {
    HostData host_data;
    host_data.csr = build_csr(adjacency_list);
    host_data.num_nodes = host_data.csr.offsets.size() - 1;
    host_data.distances.resize(host_data.num_nodes);
    std::fill(host_data.distances.begin(), host_data.distances.end(), -1);
    host_data.parents.resize(host_data.num_nodes);
    std::fill(host_data.parents.begin(), host_data.parents.end(), -1);
    return host_data;
}


std::vector<std::vector<index_type>> read_graph_from_file(std::string path) {
    std::vector<std::vector<index_type>> adjacency_list;
    size_t num_nodes;
    std::ifstream file(path);

    int num_edges = 0;
    file >> num_nodes;
    adjacency_list.resize(num_nodes);
    int src, dst;
    int i = 0;
    while (file >> src >> dst) {
        adjacency_list[src].push_back(dst);
        num_edges++;
    }
    file.close();

    return adjacency_list;
}

std::vector<std::vector<index_type>> read_undirected_graph_from_file(std::string path) {
    std::vector<std::vector<index_type>> adjacency_list;
    size_t num_nodes;

    std::ifstream file(path);
    int num_edges = 0;
    file >> num_nodes;
    adjacency_list.resize(num_nodes);
    int src, dst;
    while (file >> src >> dst) {
        adjacency_list[dst].push_back(src);
        adjacency_list[src].push_back(dst);
        num_edges++;
    }
    file.close();

    return adjacency_list;
}

#include <vector>
#include "types.hpp"

#ifndef CSR_GRAPH_HPP
#define CSR_GRAPH_HPP

typedef struct {
    std::vector<nodeid_t> offsets;
    std::vector<nodeid_t> edges;
} CSR;

typedef struct {
    size_t num_nodes;
    CSR csr;
    std::vector<nodeid_t> parents;
    std::vector<distance_t> distances;
} CSRHostData;

typedef struct {
    size_t num_nodes;
    std::vector<std::vector<adjidx_t>> adj_matrix;
} MatrixHostData;

#endif

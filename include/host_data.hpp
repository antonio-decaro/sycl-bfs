#include <vector>
#include "types.hpp"

#ifndef CSR_GRAPH_HPP
#define CSR_GRAPH_HPP

typedef struct {
    std::vector<index_type> offsets;
    std::vector<index_type> edges;
} CSR;

typedef struct {
    size_t num_nodes;
    CSR csr;
    std::vector<index_type> distances, parents;
} HostData;

#endif

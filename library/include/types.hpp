#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

// types
typedef unsigned node_t;
typedef unsigned label_t;

typedef uint32_t mask_t;
typedef unsigned char tile_t;

// constants
constexpr size_t MAX_NODES = 100000000; // get the max number of nodes in graph
constexpr size_t GLOBAL_SIZE = 1024;
constexpr size_t DEFAULT_WORK_GROUP_SIZE = 32;
const size_t MASK_SIZE = sizeof(mask_t) * 8;
const node_t INVALID_NODE = 10000;//std::numeric_limits<node_t>::max();

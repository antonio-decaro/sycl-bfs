#ifndef UTILS_TILES_HPP
#define UTILS_TILES_HPP

#include <vector>
#include "types.hpp"

enum CompressDirection {
    COMPRESS_ROW,
    COMPRESS_COL
};

template<CompressDirection direction>
char process_matrix_tile(std::vector<std::vector<char>> adj_matrix, int i, int j, int tile_size) {

    char result = 0;
    if constexpr(direction == COMPRESS_ROW) {
        for (int k = 0; k < tile_size; k++) {
            if (j * tile_size + k < adj_matrix[i].size()) {
                result |= adj_matrix[i][j * tile_size + k] << (tile_size - 1 - k);
            }
        }
    } else {
        for (int k = 0; k < tile_size; k++) {
            if (j * tile_size + k < adj_matrix.size()) {
                result |= adj_matrix[j * tile_size + k][i] << (tile_size - 1 - k);
            }
        }
    }
    return result;
}

#endif
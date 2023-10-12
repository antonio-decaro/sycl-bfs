/* 
    Inspired by the work of: 
    Haonan Ji, Huimin Song, Shibo Lu, Zhou Jin, Guangming Tan and Weifeng Liu, 
    "TileSpMSpV: A Tiled Algorithm for Sparse Matrix-Sparse Vector Multiplication on GPUs,"
    Proceedings of the 51st International Conference on Parallel Processing (ICPP), 2022, pp. 1-11, 
    DOI: https://doi.org/10.1145/3545008.3545028.
*/

#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "kernel_sizes.hpp"
#include "bfs.hpp"
#include "utils_tiles.hpp"

namespace s = sycl;

class TileData {
public:

    TileData(MatrixHostData& host_data, size_t tile_size) : host_data(host_data), tile_size(tile_size) {
        cols = host_data.num_nodes / tile_size + (host_data.num_nodes % tile_size != 0); // is the number of tiles in a row
        rows = cols * tile_size;

        tiled_matrixCSR = std::vector<std::vector<tile_t>>(rows, std::vector<tile_t>(cols, 0));
        tiled_matrixCSC = std::vector<std::vector<tile_t>>(cols, std::vector<tile_t>(rows, 0));

        constructTiledMatrices();
        constructCompressedMatrices();
    }

    MatrixHostData& host_data;
    size_t tile_size;
    size_t cols;
    size_t rows;
    std::vector<std::vector<tile_t>> tiled_matrixCSR, tiled_matrixCSC;
    std::vector<tile_t> compressed_tiled_matrixCSR, compressed_tiled_matrixCSC;
private:
    void constructTiledMatrices() {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                tiled_matrixCSR[i][j] = i < host_data.num_nodes ? 
                    process_matrix_tile<COMPRESS_ROW>(host_data.adj_matrix, i, j, tile_size) : 0;
                tiled_matrixCSC[j][i] = i < host_data.num_nodes ? 
                    process_matrix_tile<COMPRESS_COL>(host_data.adj_matrix, j, i, tile_size) : 0;
            }
        }
    }
    void constructCompressedMatrices() {
        compressed_tiled_matrixCSR.resize(rows * cols);
        for (int i = 0; i < tiled_matrixCSR.size(); i++) {
            compressed_tiled_matrixCSR.insert(
                compressed_tiled_matrixCSR.end(), 
                std::move_iterator(tiled_matrixCSR[i].begin()), 
                std::move_iterator(tiled_matrixCSR[i].end())
            );
        }
        compressed_tiled_matrixCSC.resize(rows * cols);
        for (int i = 0; i < tiled_matrixCSC.size(); i++) {
            compressed_tiled_matrixCSC.insert(
                compressed_tiled_matrixCSC.end(), 
                std::move_iterator(tiled_matrixCSC[i].begin()), 
                std::move_iterator(tiled_matrixCSC[i].end())
            );
        }
    }
};

class SYCL_TiledData {
public:
    SYCL_TiledData(TileData& tile_data) : 
        tile_data(tile_data),
        tiled_matrixCSR(s::buffer<adjidx_t, 2>{tile_data.compressed_tiled_matrixCSR.data(), s::range<2>{tile_data.rows, tile_data.cols}}),
        tiled_matrixCSC(s::buffer<adjidx_t, 2>{tile_data.compressed_tiled_matrixCSC.data(), s::range<2>{tile_data.cols, tile_data.rows}})
    {}

    TileData& tile_data;
    s::buffer<adjidx_t, 2> tiled_matrixCSR, tiled_matrixCSC;
};


void TileBFS::run() {

    // SYCL queue definition
    s::queue queue (s::gpu_selector_v, 
                    s::property_list{s::property::queue::enable_profiling{}});

    // construct data
    TileData tile_data(this->data, 4);
    SYCL_TiledData sycl_data(tile_data);


}
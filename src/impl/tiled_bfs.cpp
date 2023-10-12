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
// TO REMOVE
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <ostream>
#include <iomanip>


namespace s = sycl;

class TileData {
public:

    TileData(MatrixHostData& host_data, size_t tile_size) : host_data(host_data), tile_size(tile_size) {
        n_tiles = host_data.num_nodes / tile_size + (host_data.num_nodes % tile_size != 0);
        lenght = n_tiles * tile_size;

        tiled_matrixCSR = std::vector<std::vector<tile_t>>(lenght, std::vector<tile_t>(n_tiles, 0));
        tiled_matrixCSC = std::vector<std::vector<tile_t>>(n_tiles, std::vector<tile_t>(lenght, 0));

        constructTiledMatrices();
        constructCompressedMatrices();
    }

    MatrixHostData& host_data;
    size_t tile_size;
    size_t n_tiles;
    size_t lenght;
    std::vector<std::vector<tile_t>> tiled_matrixCSR, tiled_matrixCSC;
    std::vector<tile_t> compressed_tiled_matrixCSR, compressed_tiled_matrixCSC;
private:
    void constructTiledMatrices() {
        for (int i = 0; i < lenght; i++) {
            for (int j = 0; j < n_tiles; j++) {
                tiled_matrixCSR[i][j] = i < host_data.num_nodes ? 
                    process_matrix_tile<COMPRESS_ROW>(host_data.adj_matrix, i, j, tile_size) : 0;
                tiled_matrixCSC[j][i] = i < host_data.num_nodes ? 
                    process_matrix_tile<COMPRESS_COL>(host_data.adj_matrix, j, i, tile_size) : 0;
            }
        }
    }
    void constructCompressedMatrices() {
        compressed_tiled_matrixCSR.resize(lenght * n_tiles);
        for (int i = 0; i < tiled_matrixCSR.size(); i++) {
            compressed_tiled_matrixCSR.insert(
                compressed_tiled_matrixCSR.end(), 
                std::move_iterator(tiled_matrixCSR[i].begin()), 
                std::move_iterator(tiled_matrixCSR[i].end())
            );
        }
        compressed_tiled_matrixCSC.resize(lenght * n_tiles);
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
        tiled_matrixCSR(s::buffer<adjidx_t, 2>{tile_data.compressed_tiled_matrixCSR.data(), s::range<2>{tile_data.lenght, tile_data.n_tiles}}),
        tiled_matrixCSC(s::buffer<adjidx_t, 2>{tile_data.compressed_tiled_matrixCSC.data(), s::range<2>{tile_data.n_tiles, tile_data.lenght}})
    {}

    TileData& tile_data;
    s::buffer<adjidx_t, 2> tiled_matrixCSR, tiled_matrixCSC;
};


void TileBFS::run() {

    // SYCL queue definition
    // sycl::queue queue (sycl::gpu_selector_v, 
    //                    sycl::property_list{sycl::property::queue::enable_profiling{}});

    std::cout << "START" << std::endl;
    TileData tile_data(this->data, 4);
    std::cout << "BUILT" << std::endl;

    // print adj matrix to file
    std::ofstream file("adj_matrix.txt");
    for (int i = 0; i < tile_data.host_data.adj_matrix.size(); i++) {
        for (int j = 0; j < tile_data.host_data.adj_matrix[i].size(); j++) {

            file << (int)tile_data.host_data.adj_matrix[i][j] << " ";
        }
        file << std::endl;
    }

    // print tiles to file
    std::ofstream file2("tiles.txt");
    for (int i = 0; i < tile_data.tiled_matrixCSC.size(); i++) {
        if (!(i % tile_data.n_tiles)) file2 << std::endl;
        for (int j = 0; j < tile_data.tiled_matrixCSC[i].size(); j++) {

            file2 << std::setw(2) << std::setfill('0') << (int)tile_data.tiled_matrixCSC[i][j] << " ";
        }
        file2 << std::endl;
    }
}
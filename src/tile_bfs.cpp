#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "sycl_data.hpp"
#include "kernel_sizes.hpp"
#include "bfs.hpp"


namespace s = sycl;

class TileData {
public:

    TileData(MatrixHostData& host_data, size_t tile_size) : host_data(host_data), tile_size(tile_size) {
        int n_tiles = host_data.num_nodes / tile_size + (host_data.num_nodes % tile_size != 0);
        int size = n_tiles * tile_size;

        tiled_matrix.resize(size);
        for (auto i = 0; i < host_data.num_nodes; i++) {
            tiled_matrix[i].resize(host_data.num_nodes);
        }

        
    }

    size_t tile_size;
    MatrixHostData& host_data;
    std::vector<std::vector<char>> tiled_matrix;

private:
    void build_tiled_matrix() {

    }
};

class SYCL_TiledData {
    TileData& tile_data;

    s::buffer<char, 2> tiled_matrix;
};


void TileBFS::run() {

    // SYCL queue definition
    sycl::queue queue (sycl::gpu_selector_v, 
                       sycl::property_list{sycl::property::queue::enable_profiling{}});


}
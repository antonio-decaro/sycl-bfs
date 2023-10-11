#include <sycl/sycl.hpp>
#include "host_data.hpp"
#include "utils.hpp"
#include "arg_parse.hpp"
#include "kernel_sizes.hpp"
#include "bfs.hpp"

int main(int argc, char** argv) {

    if (!check_args(argc, argv)) {
        return 1;
    }

    // data definition
    MatrixHostData data = build_adj_matrix(argv[1]);
    std::cout << "[*] Graph loaded!" << std::endl;
    std::cout << "[*] Number of nodes: " << data.num_nodes << std::endl; 

    try {
        // run BFS
        TileBFS bfs(data);
        bfs.run();

    
    } catch (sycl::exception e) {
        std::cout << e.what() << std::endl;
    }
}
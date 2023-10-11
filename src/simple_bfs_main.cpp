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
    CSRHostData data = build_csr_host_data(read_graph_from_file(argv[1]));
    std::cout << "[*] Graph loaded!" << std::endl;
    std::cout << "[*] Number of nodes: " << data.num_nodes << std::endl; 

    try {
        // run BFS
        SimpleBFS bfs(data);
        bfs.run();

        for (nodeid_t i = 0; i < data.parents.size(); i++) {
            std::cout << i << " Parent: " << data.parents[i] << " Distance: " << data.distances[i] << std::endl;
        }
    } catch (sycl::exception e) {
        std::cout << e.what() << std::endl;
    }
}
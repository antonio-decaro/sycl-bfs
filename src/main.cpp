#include <sycl/sycl.hpp>
#include "host_data.hpp"
#include "sycl_data.hpp"
#include "utils.hpp"
#include "arg_parse.hpp"
#include "kernel_sizes.hpp"
#include "bfs.hpp"

int main(int argc, char** argv) {

    if (!check_args(argc, argv)) {
        return 1;
    }

    // data definition
    CSRHostData data = build_csr_host_data(read_adjacency_list(argv[1]));
    std::cout << "[*] Graph loaded!" << std::endl;
    std::cout << "[*] Number of nodes: " << data.num_nodes << std::endl; 
    std::cout << "[*] Offset size: " << data.csr.offsets.size() << std::endl;
    std::cout << "[*] Edges size: " << data.csr.edges.size() << std::endl;

    // init data
    data.distances[0] = 0;

    try {
        // run BFS
        SimpleBFS bfs(data);
        bfs.run();

        if (argc >= 3) {
            std::ofstream out(argv[2]);
            for (index_type i = 0; i < data.parents.size(); i++) {
                out << i << " Parent: " << data.parents[i] << " Distance: " << data.distances[i] << std::endl;
            }
            out.close();
        }
    } catch (sycl::exception e) {
        std::cout << e.what() << std::endl;
    }
}
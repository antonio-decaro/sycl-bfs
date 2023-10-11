#include <sycl/sycl.hpp>
#include "host_data.hpp"
#include "utils.hpp"
#include "arg_parse.hpp"
#include "kernel_sizes.hpp"
#include "bfs.hpp"

int main(int argc, char** argv) {

    bool print_result = false;

    // read multiple graphs from different files
    if (argc >= 3) {
        std::vector<CSRHostData> graphs;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "-p") {
                print_result = true;
                continue;
            }
            graphs.push_back(build_csr_host_data(read_graph_from_file(argv[i])));
        }
        std::cout << "[*] Graphs loaded!" << std::endl;

        // run BFS
        try {
            MultipleSimpleBFS bfs(graphs);
            bfs.run();

            if (print_result) {
                for (int i = 0; i < graphs.size(); i++) {
                    std::cout << "[!!!] Graph " << i << std::endl;
                    for (nodeid_t j = 0; j < graphs[i].parents.size(); j++) {
                        std::cout << j << " Parent: " << graphs[i].parents[j] << " Distance: " << graphs[i].distances[j] << std::endl;
                    }
                }

            }
        } catch (sycl::exception e) {
            std::cout << e.what() << std::endl;
        }
        return 0;
    }
}
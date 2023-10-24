#include <sycl/sycl.hpp>
#include "host_data.hpp"
#include "utils.hpp"
#include "arg_parse.hpp"
#include "kernel_sizes.hpp"
#include "bfs.hpp"
#include "benchmark.hpp"

int main(int argc, char** argv) {

    bool print_result = false;
    size_t local_size = DEFAULT_WORK_GROUP_SIZE;

    // read multiple graphs from different files
    if (argc >= 2) {
        std::vector<CSRHostData> graphs;
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "-p") {
                print_result = true;
                continue;
            } else if (std::string(argv[i]).find("-local=") == 0) {
                local_size = std::stoi(std::string(argv[i]).substr(7));
                continue;
            }
            graphs.push_back(build_csr_host_data(read_graph_from_file(argv[i])));
        }
        std::cout << "[*] Graphs loaded!" << std::endl;

        // run BFS
        try {
            MultipleGraphBFS bfs8(graphs, std::make_shared<FrontierMBFSOperator<8>>());
            MultipleGraphBFS bfs16(graphs, std::make_shared<FrontierMBFSOperator<16>>());
            MultipleGraphBFS bfs32(graphs, std::make_shared<FrontierMBFSOperator<32>>());
            
            bfs8.run(local_size); // dummy kernel

            std::cout << "SubGroup size  8:" << std::endl;
            bench_time_t time = bfs8.run(local_size);
            std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
            std::cout << "- Total time: " << time.total_time << " us" << std::endl;

            std::cout << "SubGroup size 16:" << std::endl;
            time = bfs16.run(local_size);
            std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
            std::cout << "- Total time: " << time.total_time << " us" << std::endl;
            
            std::cout << "SubGroup size 32:" << std::endl;
            time = bfs32.run(local_size);
            std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
            std::cout << "- Total time: " << time.total_time << " us" << std::endl;

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
#include <sycl/sycl.hpp>
#include <memory>
#include "host_data.hpp"
#include "utils.hpp"
#include "arg_parse.hpp"
#include "kernel_sizes.hpp"
#include "impl/simpl_bfs.hpp"
#include "operators.hpp"

int main(int argc, char **argv)
{

	if (!check_args(argc, argv))
	{
		return 1;
	}

	// data definition
	CSRHostData data = build_csr_host_data(read_graph_from_file(argv[1]));
	std::cout << "[*] Graph loaded!" << std::endl;
	std::cout << "[*] Number of nodes: " << data.num_nodes << std::endl;
	std::cout << "[*] Offset size: " << data.csr.offsets.size() << std::endl;
	std::cout << "[*] Edges size: " << data.csr.edges.size() << std::endl;

	// init data
	data.distances[0] = 0;

	try
	{
		// run BFS
		SingleBFS bfs(data, std::make_shared<FrontierBFSOperator<16>>());
		auto ret = bfs.run();
		std::cout << "[*] Kernel time: " << ret.kernel_time << " us" << std::endl;
		std::cout << "[*] Total time: " << ret.total_time << " us" << std::endl;

		if (argc >= 3)
		{
			std::ofstream out(argv[2]);
			for (nodeid_t i = 0; i < data.parents.size(); i++)
			{
				out << i << " Parent: " << data.parents[i] << " Distance: " << data.distances[i] << std::endl;
			}
			out.close();
		}
	}
	catch (sycl::exception e)
	{
		std::cout << e.what() << std::endl;
	}
}
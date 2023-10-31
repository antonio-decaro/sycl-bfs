#include <sycl/sycl.hpp>
#include <iomanip>
#include "host_data.hpp"
#include "utils.hpp"
#include "arg_parse.hpp"
#include "kernel_sizes.hpp"
#include "bfs.hpp"
#include "benchmark.hpp"

int main(int argc, char **argv)
{
	args_t args;
	get_mul_graph_args(argc, argv, args);

	if (args.fnames.empty())
	{
		std::cout << "[!] No graph to process!" << std::endl;
		return 0;
	}

	std::cout << "[*] " << args.graphs.size() << " Graphs loaded!" << std::endl;

	std::vector<nodeid_t> sources;
	for (int i = 0; i < args.graphs.size(); i++)
	{
		sources.push_back(0);
	}

	// run BFS
	try
	{
		MultipleGraphBFS bfs8(args.graphs, std::make_shared<FrontierMBFSOperator<8>>());
		MultipleGraphBFS bfs16(args.graphs, std::make_shared<FrontierMBFSOperator<16>>());
		MultipleGraphBFS bfs32(args.graphs, std::make_shared<FrontierMBFSOperator<32>>());
		bench_time_t time;

		std::cout << "SubGroup size  8:" << std::endl;
		time = bfs8.run(sources.data(), args.local_size);
		std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
		std::cout << "- Total time: " << time.total_time << " us" << std::endl;

		std::cout << "SubGroup size 16:" << std::endl;
		time = bfs16.run(sources.data(), args.local_size);
		std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
		std::cout << "- Total time: " << time.total_time << " us" << std::endl;

		std::cout << "SubGroup size 32:" << std::endl;
		time = bfs32.run(sources.data(), args.local_size);
		std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
		std::cout << "- Total time: " << time.total_time << " us" << std::endl;

		if (args.print_result)
		{
			for (int i = 0; i < args.graphs.size(); i++)
			{
				std::cout << "[!!!] Graph " << i << ": " << args.fnames[i] << std::endl;
				for (nodeid_t j = 0; j < args.graphs[i].num_nodes; j++)
				{
					std::cout << "- N: " << std::setfill(' ') << std::setw(3) << j
										<< " | Parent: " << std::setfill(' ') << std::setw(3) << args.graphs[i].parents[j] 
										<< " | Distance: " << std::setfill(' ') << std::setw(2) << args.graphs[i].distances[j] << std::endl;
				}
			}
		}
	}
	catch (sycl::exception e)
	{
		std::cout << e.what() << std::endl;
	}
	return 0;
}
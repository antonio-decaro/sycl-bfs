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

	bool print_result = false;
	size_t local_size = DEFAULT_WORK_GROUP_SIZE;
	std::vector<std::string> fnames;
	std::vector<CSRHostData> graphs;
	std::string directory = "";

	// read multiple graphs from different files
	if (argc >= 2)
	{
		for (int i = 1; i < argc; i++)
		{
			if (std::string(argv[i]) == "-p")
			{
				print_result = true;
				continue;
			}
			else if (std::string(argv[i]).find("-local=") == 0)
			{
				local_size = std::stoi(std::string(argv[i]).substr(7));
				continue;
			} else if (std::string(argv[i]).find("-d=") == 0) 
			{
				directory = std::string(argv[i]).substr(3);
				continue;
			}
			fnames.push_back(argv[i]);
		}
	}

	if (directory != "")
	{
		fnames = get_files_in_directory(directory);
	}

	if (fnames.empty())
	{
		std::cout << "[!] No graph to process!" << std::endl;
		return 0;
	}

	for (auto &s : fnames)
	{
		graphs.push_back(build_csr_host_data(read_graph_from_file(s)));
	}
	std::cout << "[*] " << graphs.size() << " Graphs loaded!" << std::endl;

	std::vector<nodeid_t> sources;
	for (int i = 0; i < graphs.size(); i++)
	{
		sources.push_back(0);
	}

	// run BFS
	try
	{
		MultipleGraphBFS<false> bfs8(graphs, std::make_shared<FrontierMBFSOperator<8>>());
		MultipleGraphBFS<false> bfs16(graphs, std::make_shared<FrontierMBFSOperator<16>>());
		MultipleGraphBFS<false> bfs32(graphs, std::make_shared<FrontierMBFSOperator<32>>());


		std::cout << "SubGroup size  8:" << std::endl;
		bfs8.run(sources.data(), local_size); // dummy kernel
		bench_time_t time = bfs8.run(sources.data(), local_size);
		std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
		std::cout << "- Total time: " << time.total_time << " us" << std::endl;

		std::cout << "SubGroup size 16:" << std::endl;
		bfs16.run(sources.data(), local_size); // dummy kernel
		time = bfs16.run(sources.data(), local_size);
		std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
		std::cout << "- Total time: " << time.total_time << " us" << std::endl;

		std::cout << "SubGroup size 32:" << std::endl;
		bfs32.run(sources.data(), local_size); // dummy kernel
		time = bfs32.run(sources.data(), local_size);
		std::cout << "- Kernel time: " << time.kernel_time << " us" << std::endl;
		std::cout << "- Total time: " << time.total_time << " us" << std::endl;

		if (print_result)
		{
			for (int i = 0; i < graphs.size(); i++)
			{
				std::cout << "[!!!] Graph " << i << std::endl;
				for (nodeid_t j = 0; j < graphs[i].num_nodes; j++)
				{
					std::cout << "- N: " << std::setfill(' ') << std::setw(3) << j
										<< " | Parent: " << std::setfill(' ') << std::setw(3) << graphs[i].parents[j] 
										<< " | Distance: " << std::setfill(' ') << std::setw(2) << graphs[i].distances[j] << std::endl;
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
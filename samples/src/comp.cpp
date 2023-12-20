#include <sycl/sycl.hpp>
#include <iomanip>
#include "sygraph.hpp"
#include "arg_parse.hpp"

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
	sygraph::details::CompressedGraphs graphs(args.graphs);
  sygraph::SYGraphsCompressed sycl_graphs(graphs);

	std::vector<node_t> sources;
	for (int i = 0; i < args.graphs.size(); i++)
	{
		sources.push_back(0);
	}

  std::chrono::_V2::system_clock::time_point start, end;
  std::chrono::microseconds time;

	// run BFS
	try
	{
    sycl::queue q {sycl::gpu_selector_v};
		sygraph::BFSCompressed bfs(q, sycl_graphs);

		bfs.run(args.local_size, sources);
		bfs.wait();

#ifdef SUPPORTS_SG_8
		std::cout << "SubGroup size  8:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
		bfs.run<8>(args.local_size, sources);
		bfs.wait();
		end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "- Total time: " << time.count() << " us" << std::endl;
#endif

		std::cout << "SubGroup size 16:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
		bfs.run<16>(args.local_size, sources);
    bfs.wait();
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "- Total time: " << time.count() << " us" << std::endl;

		std::cout << "SubGroup size 32:" << std::endl;
    start = std::chrono::high_resolution_clock::now();
		bfs.run<32>(args.local_size, sources);
		bfs.wait();
    end = std::chrono::high_resolution_clock::now();
    time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
		std::cout << "- Total time: " << time.count() << " us" << std::endl;

		q.wait_and_throw();
		if (args.print_result)
		{
			auto accessor = bfs.getParentsHostAccessor();
			int i = 0;
			for (int j = 0; j < args.graphs.size(); i++)
			{
				std::cout << "[!!!] Graph " << j << ": " << args.fnames[j] << std::endl;
				for (int k = 0; k < args.graphs[j].getSize(); k++, i++)
				{
					std::cout << "- Node: " << std::setfill(' ') << std::setw(3) << k
										<< " | Parent: " << std::setfill(' ') << std::setw(3) << accessor[i] << std::endl;
				}
			}
		}
	}
	catch (sycl::exception e)
	{
		std::cerr << e.what() << std::endl;
	} catch (std::exception e) {
		std::cerr << e.what() << std::endl;
	} catch (...) {
		std::cerr << "Unknown error" << std::endl;
	}
	return 0;
}
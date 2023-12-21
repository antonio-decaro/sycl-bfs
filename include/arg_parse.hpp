#include <fstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include "kernel_sizes.hpp"
#include "types.hpp"
#include "host_data.hpp"
#include "utils.hpp"

// read the graph from the file
bool check_args(int &argc, char **&argv)
{
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " <graph_path> [out_data]" << std::endl;
		return false;
	}
	return true;
}

std::vector<std::string> get_files_in_directory(const std::string dname) {
	std::vector<std::string> fnames;
	for (const auto & entry : std::filesystem::directory_iterator(dname)) {
		fnames.push_back(entry.path().string());
	}
	return fnames;
}

typedef struct {
	bool print_result = false;
	size_t local_size;
	std::vector<std::string> fnames;
	std::vector<CSRHostData> graphs;
} args_t;

void get_mul_graph_args(int argc, char** argv, args_t &args, bool undirected = false) {
	std::string directory = "";
	std::vector<std::string> tmp_fnames;
	args.local_size = DEFAULT_WORK_GROUP_SIZE;

	if (argc >= 2)
	{
		for (int i = 1; i < argc; i++)
		{
			if (std::string(argv[i]) == "-p")
			{
				args.print_result = true;
				continue;
			}
			else if (std::string(argv[i]).find("-local=") == 0)
			{
				args.local_size = std::stoi(std::string(argv[i]).substr(7));
				continue;
			} else if (std::string(argv[i]).find("-d=") == 0) {
				directory = std::string(argv[i]).substr(3);
				continue;
			} else if (std::string(argv[i]).find("-h") != std::string::npos || std::string(argv[i]).find("--help") != std::string::npos) {
				std::cout << "Usage: " << argv[0] << " [-p] [-local=<local_size>] <graph files or directories...>" << std::endl;
				exit(0);
			}
			tmp_fnames.push_back(argv[i]);
		}
	}

	for (auto &s : tmp_fnames)
	{
		if (std::filesystem::is_directory(s)) {
			auto files = get_files_in_directory(s);
			for (auto &f : files) {
				args.fnames.push_back(f);
			}
		} else {
			args.fnames.push_back(s);
		}
	}

	for (auto &s : args.fnames)
	{
		args.graphs.push_back(readGraphFromFile(s, false));
	}
}
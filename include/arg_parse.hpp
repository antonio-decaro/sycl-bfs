#include <fstream>
#include <iostream>

// read the graph from the file
bool check_args(int& argc, char**& argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <graph_path> [out_data]" << std::endl;
        return false;
    }
    return true;
}
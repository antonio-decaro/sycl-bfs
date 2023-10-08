#include <sycl/sycl.hpp>
#include "sycl_data.hpp"
#include "host_data.hpp"

class SimpleBFS {
private:
    CSRHostData& data;

public:
    SimpleBFS(CSRHostData& data) : 
        data(data) {}

    void run();

    std::vector<nodeid_t> get_result() { return data.parents; }
};

class TileBFS {
private:
    MatrixHostData& data;

public:
    TileBFS(MatrixHostData& data) : 
        data(data) {}
    
    void run();
};
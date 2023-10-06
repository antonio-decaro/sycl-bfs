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

    std::vector<index_type> get_result() { return data.parents; }
};

class TileBFS {
private:
    CSRHostData& data;

public:
    TileBFS(CSRHostData& data) : 
        data(data) {}
    
    void run();
};
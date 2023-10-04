#include <sycl/sycl.hpp>
#include "sycl_data.hpp"
#include "host_data.hpp"

class SimpleBFS {
private:
    HostData& data;

public:
    SimpleBFS(HostData& data) : 
        data(data) {}

    void run();

    std::vector<index_type> get_result() { return data.parents; }
};
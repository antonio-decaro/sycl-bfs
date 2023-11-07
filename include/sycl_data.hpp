#include <sycl/sycl.hpp>
#include <vector>
#include <string>
#include "host_data.hpp"
#include "kernel_sizes.hpp"
#include "types.hpp"

#ifndef __SYCL_DATA_HPP__
#define __SYCL_DATA_HPP__

class SYCL_VectorizedGraphData
{
public:
	SYCL_VectorizedGraphData(std::vector<CSRHostData> &data) : data(data)
	{

		for (auto &d : data)
		{
			offsets.push_back(sycl::buffer<size_t, 1>{d.csr.offsets.data(), sycl::range{d.csr.offsets.size()}});
			edges.push_back(sycl::buffer<nodeid_t, 1>{d.csr.edges.data(), sycl::range{d.csr.edges.size()}});
			parents.push_back(sycl::buffer<nodeid_t, 1>{d.parents.data(), sycl::range{d.parents.size()}});
		}
	}

	sycl::event init(sycl::queue &q, const std::vector<nodeid_t> &sources, size_t wg_size = DEFAULT_WORK_GROUP_SIZE)
	{
		size_t num_graphs = data.size();
		sycl::buffer<nodeid_t, 1> device_source{sources.data(), sycl::range{sources.size()}};

		sycl::accessor<size_t, 1, sycl::access::mode::read> offsets_acc[MAX_PARALLEL_GRAPHS];
		sycl::accessor<nodeid_t, 1, sycl::access::mode::read> edges_acc[MAX_PARALLEL_GRAPHS];
		sycl::accessor<nodeid_t, 1, sycl::access::mode::discard_read_write> parents_acc[MAX_PARALLEL_GRAPHS];

		return q.submit([&](sycl::handler &cgh) {
			size_t n_nodes [MAX_PARALLEL_GRAPHS];

			sycl::range global {num_graphs * wg_size};
			sycl::range local {wg_size};

      for (int i = 0; i < num_graphs; i++) {
        offsets_acc[i] = offsets[i].get_access<sycl::access::mode::read>(cgh);
        edges_acc[i] = edges[i].get_access<sycl::access::mode::read>(cgh);
        parents_acc[i] = parents[i].get_access<sycl::access::mode::discard_read_write>(cgh);
        n_nodes[i] = data[i].num_nodes;
      } 

			sycl::accessor sources{device_source, cgh, sycl::read_only};

			cgh.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
				auto gid = item.get_group_linear_id();
				auto lid = item.get_local_linear_id();
				auto local_range = item.get_local_range(0);
				auto nodes_count = n_nodes[gid];
				auto source = sources[gid];

				auto pp = parents_acc[gid];

				for (int i = lid; i < nodes_count; i += local_range) {
					pp[i] = -1;
					if (i == source) {
						pp[i] = source;
					}
				}
			});
		});
	}

	void write_back()
	{
		for (int i = 0; i < data.size(); i++)
		{
			auto &d = data[i];
			parents[i].set_final_data(d.parents.data());
			parents[i].set_write_back(true);
		}
	}

	std::vector<CSRHostData> &data;
	std::vector<sycl::buffer<size_t, 1>> offsets;
	std::vector<sycl::buffer<nodeid_t, 1>> edges;
	std::vector<sycl::buffer<nodeid_t, 1>> parents;
};


class SYCL_CompressedGraphData
{
public:
	SYCL_CompressedGraphData(CompressedHostData &data) : 
		host_data(data),
		nodes_offsets(sycl::buffer<size_t, 1>(data.nodes_offsets.data(), sycl::range{data.nodes_offsets.size()})),
		graphs_offests(sycl::buffer<size_t, 1>(data.graphs_offsets.data(), sycl::range{data.graphs_offsets.size()})),
		nodes_count(sycl::buffer<size_t, 1>(data.nodes_count.data(), sycl::range{data.nodes_count.size()})),
		edges_offsets(sycl::buffer<size_t, 1>{data.compressed_offsets.data(), sycl::range{data.compressed_offsets.size()}}),
		edges(sycl::buffer<nodeid_t, 1>{data.compressed_edges.data(), sycl::range{data.compressed_edges.size()}}),
		parents(sycl::buffer<nodeid_t, 1>{data.compressed_parents.data(), sycl::range{data.compressed_parents.size()}}) {}

	sycl::event init(sycl::queue &q, const std::vector<nodeid_t> &sources, size_t wg_size = DEFAULT_WORK_GROUP_SIZE)
	{
		sycl::buffer<nodeid_t, 1> device_source{sources.data(), sycl::range{sources.size()}};

		return q.submit([&](sycl::handler &h) {
			sycl::range global {host_data.num_graphs * wg_size};
			sycl::range local {wg_size};

			sycl::accessor sources {device_source, h, sycl::read_only};
			sycl::accessor nodes_acc{nodes_offsets, h, sycl::read_only};
			sycl::accessor nodes_count_acc{nodes_count, h, sycl::read_only};
			sycl::accessor parents_acc{parents, h, sycl::write_only, sycl::no_init};

			h.parallel_for(sycl::nd_range<1>{global, local}, [=](sycl::nd_item<1> item) {
				auto gid = item.get_group_linear_id();
				auto lid = item.get_local_linear_id();
				auto local_range = item.get_local_range(0);
				auto nodes_count = nodes_count_acc[gid];
				auto source = sources[gid];

				for (int i = lid; i < nodes_count; i += local_range) {
					parents_acc[nodes_acc[gid] + i] = -1;
					if (i == source) {
						parents_acc[nodes_acc[gid] + i] = source;
					}
				}
			}); 
		});
	}

	void write_back()
	{
		auto pacc = parents.get_host_access();
		for (int i = 0; i < host_data.compressed_parents.size(); i++)
		{
			host_data.compressed_parents[i] = pacc[i];
		}

		host_data.write_back();
	}

	CompressedHostData &host_data;
	sycl::buffer<nodeid_t, 1> edges, parents;
	sycl::buffer<size_t, 1> graphs_offests, nodes_offsets, nodes_count, edges_offsets;
};

class SYCL_SimpleGraphData
{
public:
	SYCL_SimpleGraphData(CSRHostData &data) : 
		host_data(data),
		num_nodes(data.num_nodes),
		edges_offsets(sycl::buffer<size_t, 1>{data.csr.offsets.data(), sycl::range{data.csr.offsets.size()}}),
		edges(sycl::buffer<nodeid_t, 1>{data.csr.edges.data(), sycl::range{data.csr.edges.size()}}),
		parents(sycl::buffer<nodeid_t, 1>{data.parents.data(), sycl::range{data.parents.size()}})
	{
		parents.set_write_back(false);
	}

	void write_back()
	{
		parents.set_final_data(host_data.parents.data());
		parents.set_write_back(true);
	}

	sycl::event init(sycl::queue &q, const nodeid_t source) {
		return q.submit([&](sycl::handler &h) {
			sycl::accessor parents_acc{parents, h, sycl::write_only, sycl::no_init};

			h.parallel_for(sycl::range<1>{num_nodes}, [=](sycl::id<1> idx) {
				int node = idx[0];
				parents_acc[node] = -1;
				if (node == source) {
					parents_acc[node] = node;
				}
			});
		});
	}

	size_t num_nodes;
	CSRHostData &host_data;
	sycl::buffer<nodeid_t, 1> parents, edges;
	sycl::buffer<size_t, 1> edges_offsets;
};

#endif
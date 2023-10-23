#ifndef __SIMPL_BFS_HPP__
#define __SIMPL_BFS_HPP__

#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include "host_data.hpp"
#include "kernel_sizes.hpp"
#include "sycl_data.hpp"
#include "benchmark.hpp"

namespace s = sycl;

class SingleBFSOperator
{
public:
	virtual void operator()(sycl::queue &queue, SYCL_SimpleGraphData &data, std::vector<sycl::event> &events) = 0;
};

void multi_events_BFS(sycl::queue &queue, SYCL_SimpleGraphData &data, std::vector<sycl::event> &events)
{

	bool *changed = s::malloc_shared<bool>(1, queue);
	queue.fill(changed, false, 1);
	int level = 0;

	do
	{
		*changed = false;
		auto e = queue.submit([&](sycl::handler &h)
													{
            s::accessor offsets_acc(data.edges_offsets, h, s::read_only);
            s::accessor edges_acc(data.edges, h, s::read_only);
            s::accessor distances_acc(data.distances, h, s::read_write);
            s::accessor parents_acc(data.parents, h, s::write_only, s::no_init);

            h.parallel_for(s::range<1>{data.num_nodes}, [=, num_nodes=data.num_nodes](s::id<1> idx) {
                int node = idx[0];
                if (distances_acc[node] == level) {
                    for (int i = offsets_acc[node]; i < offsets_acc[node + 1]; i++) {
                        int neighbor = edges_acc[i];
                        if (distances_acc[neighbor] == -1) {
                            distances_acc[neighbor] = level + 1;
                            parents_acc[neighbor] = node;
                            *changed = true;
                        }
                    }
                }
            }); });
		events.push_back(e);
		e.wait();
		level++;
	} while (*changed);

	std::cout << "[*] Max depth reached: " << level << std::endl;
}

class SingleBFS
{
private:
	CSRHostData &data;
	std::shared_ptr<SingleBFSOperator> op;

public:
	SingleBFS(CSRHostData &data, std::shared_ptr<SingleBFSOperator> op) : data(data), op(op) {}

	bench_time_t run() {
		// SYCL queue definition
		sycl::queue queue(sycl::gpu_selector_v,
											sycl::property_list{sycl::property::queue::enable_profiling{}});

		SYCL_SimpleGraphData sycl_data(data);
		std::vector<sycl::event> events;

		auto start_glob = std::chrono::high_resolution_clock::now();
		(*op)(queue, sycl_data, events);
		auto end_glob = std::chrono::high_resolution_clock::now();

		long duration = 0;
		for (sycl::event &e : events) {
			auto start = e.get_profiling_info<sycl::info::event_profiling::command_start>();
			auto end = e.get_profiling_info<sycl::info::event_profiling::command_end>();
			duration += (end - start);
		}

		sycl_data.write_back();

		return bench_time_t {
			.kernel_time = static_cast<float>(duration) / 1000,
			.total_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count()),
			.to_microsec = 1.0f
		};
	}
};

#endif
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

// TODO fix: this doesn't work after removing distances

class SingleBFSOperator
{
public:
	virtual void operator()(sycl::queue &queue, SYCL_SimpleGraphData &data, std::vector<sycl::event> &events) = 0;
};

class SingleBFS
{
private:
	CSRHostData &data;
	std::shared_ptr<SingleBFSOperator> op;

public:
	SingleBFS(CSRHostData &data, std::shared_ptr<SingleBFSOperator> op) : data(data), op(op) {}

	bench_time_t run(nodeid_t source = 0) {
		// SYCL queue definition
		sycl::queue queue(sycl::gpu_selector_v,
											sycl::property_list{sycl::property::queue::enable_profiling{}});

		SYCL_SimpleGraphData sycl_data(data);
		sycl_data.init(queue, source);
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
#ifndef MUL_BFS_HPP
#define MUL_BFS_HPP

#include <sycl/sycl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <array>
#include <memory>
#include "kernel_sizes.hpp"
#include "host_data.hpp"
#include "sycl_data.hpp"
#include "types.hpp"
#include "benchmark.hpp"

namespace s = sycl;

class MultiBFSOperator {
public:
	virtual void operator() (s::queue& queue, SYCL_CompressedGraphData& data, std::vector<s::event>& events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE) = 0;
	virtual void operator() (s::queue& queue, SYCL_VectorizedGraphData& data, std::vector<s::event>& events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE) = 0;
};

class MultipleGraphBFS {
public:
	MultipleGraphBFS(std::vector<CSRHostData>& data, std::shared_ptr<MultiBFSOperator> op) : 
		data(data), op(op) {}

	bench_time_t run(const size_t wg_size = DEFAULT_WORK_GROUP_SIZE, bool write_back = true) {
			// s queue definition
		s::queue queue (s::gpu_selector_v, 
						s::property_list{s::property::queue::enable_profiling{}});

		CompressedHostData compressed_data(data);
		SYCL_CompressedGraphData sycl_data(compressed_data);
		SYCL_VectorizedGraphData sycl_vectorized_data(data);

		std::vector<s::event> events;

		auto start_glob = std::chrono::high_resolution_clock::now();
		(*op)(queue, sycl_data, events, wg_size);
		auto end_glob = std::chrono::high_resolution_clock::now();

		long duration = 0;
		for (s::event& e : events) {
			auto start = e.get_profiling_info<s::info::event_profiling::command_start>();
			auto end = e.get_profiling_info<s::info::event_profiling::command_end>();
			duration += (end - start);
		}

		if (write_back) sycl_data.write_back();

		return bench_time_t {
			.total_time = static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(end_glob - start_glob).count()),
			.kernel_time = static_cast<float>(duration) / 1000,
			.to_microsec = 1.0f
		};
	}

private:
	std::vector<CSRHostData>& data;
	std::shared_ptr<MultiBFSOperator> op;
};

#endif
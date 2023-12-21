#pragma once

#include <sycl/sycl.hpp>
#include "types.hpp"
#include "sycl_data.hpp"
#include "utils.hpp"

namespace sygraph {
namespace details {
namespace s = sycl;

class BFS_Instance {
private:
  sycl::queue& queue;
  SYGraph& graph;
  size_t graph_size;
  const node_t source_node;
  sycl::buffer<node_t, 1> parents_buf;
  sycl::event waiting_event;
  bool written_back = false;

public:

  BFS_Instance() = delete;
  BFS_Instance(sycl::queue& queue, SYGraph& graph, node_t source_node, std::vector<node_t>& parents)://, std::vector<size_t> offsets, std::vector<node_t> edges, node_t source_node) : 
    queue{queue},
    source_node{source_node}, 
    // graph{offsets, edges}, 
    graph{graph}, 
    parents_buf{sycl::buffer<node_t, 1>(parents.data(), parents.size())}
  {
    graph_size = graph.getSize();
    parents_buf.set_final_data(parents.data());
    parents_buf.set_write_back(true);
  }

  template<size_t sg_size = 16>
  void run(size_t parallel_items) {
    size_t source = this->source_node;
    size_t node_count = this->graph_size;
    waiting_event = queue.submit([&, parallel_items, source, node_count](sycl::handler& cgh) {
      s::range<1> global{parallel_items};
      s::range<1> local{parallel_items};
      
      const size_t NUM_MASKS = node_count / MASK_SIZE + 1; // the number of masks needed to represent all nodes
      
      auto offsets_acc = graph.getRowOffsetsDeviceAccessor(cgh);
      auto edges_acc = graph.getColIndicesDeviceAccessor(cgh);
      s::accessor parents_acc{parents_buf, cgh, s::read_write, s::no_init};

      s::local_accessor<mask_t, 1> frontier{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> next{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> running{s::range<1>{1}, cgh};

      sycl::stream out(2048, 128, cgh);

      cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
        s::atomic_ref<mask_t, s::memory_order::relaxed, s::memory_scope::work_group, s::access::address_space::local_space> running_ar{running[0]};
        auto loc_id = item.get_local_id(0);
        auto local_size = item.get_local_range(0);

        // init the frontier
        if (loc_id == 0) {
          running_ar.store(1);
          size_t source_offset = source / MASK_SIZE;
          mask_t source_bit = 1 << (source % MASK_SIZE);
          frontier[source_offset] = next[source_offset] = source_bit;
        }
        for (int i = loc_id; i < node_count; i += local_size) {
          if (i == source) {
            parents_acc[source] = source;
          } else {
            parents_acc[i] = INVALID_NODE;
          }
        }

        s::group_barrier(item.get_group());
        while (running_ar.load()) {
          if (loc_id < NUM_MASKS) {
            frontier[loc_id] = next[loc_id];
            next[loc_id] = 0;
          }
          s::group_barrier(item.get_group());


          for (node_t node_id = loc_id; node_id < node_count; node_id += local_size) {
            int node_mask_offet = node_id / MASK_SIZE; // to access the right mask
            mask_t node_bit = 1 << (node_id % MASK_SIZE); // to access the right bit in the mask 
            s::atomic_ref<mask_t, s::memory_order::relaxed, s::memory_scope::work_group, s::access::address_space::local_space> next_ar{next[node_mask_offet]};
            if (parents_acc[node_id] == INVALID_NODE) {

              for (int i = offsets_acc[node_id]; i < offsets_acc[node_id + 1]; i++) {
                node_t neighbor = edges_acc[i];
                int neighbor_mask_offset = neighbor / MASK_SIZE;
                mask_t neighbor_bit = 1 << (neighbor % MASK_SIZE);
                if (frontier[neighbor_mask_offset] & neighbor_bit) {
                  parents_acc[node_id] = neighbor;
                  next_ar |= node_bit;
                  break;
                }
              }
            }
          }
          
          if (loc_id == 0) running_ar.store(0);
          s::group_barrier(item.get_group());
          if (loc_id < NUM_MASKS) {
            running_ar += next[loc_id];
          }
          s::group_barrier(item.get_group());
        }
      });
    });
  }

  inline sycl::event getEvent() { return waiting_event; }

  inline sycl::buffer<node_t, 1>& getParentsSYCLBuffer() { return parents_buf; }

  inline void wait() {
    waiting_event.wait_and_throw();
    writeBack();
  }

  inline const bool isWrittenBack() const noexcept { return written_back; }

  void writeBack() {
    // auto acc = parents_buf.get_host_access();
    // for (size_t i = 0; i < parents.size(); i++) {
    //   parents[i] = acc[i];
    // }
    // written_back = true;
  }
};
} // namespace details

class BFS {
private:
  sycl::queue& queue;
  std::vector<details::BFS_Instance> instances;
  std::vector<std::vector<node_t>> parents;

public:
  BFS(sycl::queue& queue, std::vector<SYGraph>& graphs) : queue(queue) {
    for (auto& graph : graphs) {
      parents.push_back(std::vector<node_t>(graph.getSize()));
      instances.emplace_back(queue, graph, 0, parents.back());
    }
  }

  template<size_t sg_size = 16>
  void run(size_t parallel_items) {
    for (auto& instance : instances) {
      instance.run<sg_size>(parallel_items);
    }
  }

  void wait() {
    // queue.wait_and_throw();
    for (auto& instance : instances) {
      instance.wait();
    }
  }

  std::vector<sycl::event> getEvents() {
    std::vector<sycl::event> events;
    for (auto& instance : instances) {
      events.push_back(instance.getEvent());
    }
    return events;
  }

  inline sycl::buffer<node_t, 1>& getParentsBuf(size_t idx) {
    if (idx >= instances.size()) throw std::out_of_range("Index out of range");
    return instances[idx].getParentsSYCLBuffer();
  }
  sycl::host_accessor<node_t, 1, sycl::access_mode::read> getParentsHostAccessor(size_t idx) {
    if (idx >= instances.size()) throw std::out_of_range("Index out of range");
    return instances[idx].getParentsSYCLBuffer().get_host_access(sycl::read_only);
  }
  const std::vector<node_t>& getParents(size_t idx) const {
    if (idx >= instances.size()) throw std::out_of_range("Index out of range");
    return parents[idx];
  }
};

} // namespace sygraph
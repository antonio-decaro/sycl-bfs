/**
 * @file bottomup_op.hpp
 * @brief Defines the BottomUpMBFSOperator class, which implements the bottom-up BFS traversal algorithm.
 */

#ifndef __BOTTOM_UP_OP_HPP__
#define __BOTTOM_UP_OP_HPP__

#include "impl/mul_bfs.hpp"

namespace s = sycl;

typedef uint32_t mask_t;
constexpr size_t MASK_SIZE = 32; // the size of the mask according to the type of mask_t

/**
 * @brief Implements the bottom-up BFS traversal algorithm.
 * 
 * This class provides two operator() overloads, one for SYCL_CompressedGraphData and one for SYCL_VectorizedGraphData.
 * Both overloads take a SYCL queue, a graph data structure, a vector of source nodes, a vector of events, and an optional work group size.
 * The operator() overloads launch a SYCL kernel that performs the bottom-up BFS traversal algorithm on the input graph(s).
 * 
 * @tparam sg_size The sub-group size to use in the kernel.
 */
template <size_t sg_size = 16>
class BottomUpMBFSOperator : public MultiBFSOperator
{
  /**
   * @brief This method performs the BFS on multiple graphs using a bottom-up approach.
   * 
   * @param queue The SYCL queue to submit the kernel to.
   * @param data The compressed graph data.
   * @param sources The vector of source nodes.
   * @param events The vector of events to be updated with the new event.
   * @param wg_size The size of the work-group to be used in the kernel.
   */
  void operator()(s::queue &queue, SYCL_CompressedGraphData &data, const std::vector<nodeid_t> &sources, std::vector<s::event> &events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE)
  {
    s::range<1> global{wg_size * (data.host_data.num_graphs)}; // each workgroup will process a graph
    s::range<1> local{wg_size};

    s::buffer<nodeid_t, 1> sources_buf{sources.data(), s::range<1>{sources.size()}};

    auto e = queue.submit([&](s::handler &cgh) {
      s::accessor offsets_acc{data.edges_offsets, cgh, s::read_only};
      s::accessor edges_acc{data.edges, cgh, s::read_only};
      s::accessor parents_acc{data.parents, cgh, s::read_write};
      s::accessor graphs_offsets_acc{data.graphs_offests, cgh, s::read_only};
      s::accessor nodes_offsets_acc{data.nodes_offsets, cgh, s::read_only};
      s::accessor nodes_count_acc{data.nodes_count, cgh, s::read_only};
      s::accessor sources_acc{sources_buf, cgh, s::read_only};

      const size_t MAX_NODES = *std::max_element(data.host_data.nodes_count.begin(), data.host_data.nodes_count.end()); // get the max number of nodes in graph
      const size_t NUM_MASKS = MAX_NODES / MASK_SIZE + 1; // the number of masks needed to represent all nodes
      s::local_accessor<mask_t, 1> frontier{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> next{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> running{s::range<1>{1}, cgh};

      cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
        s::atomic_ref<mask_t, s::memory_order::relaxed, s::memory_scope::work_group, s::access::address_space::local_space> running_ar{running[0]};
        auto grp_id = item.get_group_linear_id();
        auto loc_id = item.get_local_id(0);
        auto node_offset = nodes_offsets_acc[grp_id];
        auto node_count = nodes_count_acc[grp_id];
        auto local_size = item.get_local_range(0);

        // init the frontier
        if (loc_id == 0) {
          running_ar.store(1);
          auto source = sources_acc[grp_id];
          int source_offset = source / MASK_SIZE;
          mask_t source_bit = 1 << (source % MASK_SIZE);
          frontier[source_offset] = next[source_offset] = source_bit;
        }

        item.barrier(s::access::fence_space::local_space);
        while (running_ar.load()) {
          if (loc_id < NUM_MASKS) {
            frontier[loc_id] = next[loc_id];
            next[loc_id] = 0;
          }
          item.barrier(s::access::fence_space::local_space);

          for (nodeid_t node_id = loc_id; node_id < node_count; node_id += local_size) {
            int node_mask_offet = node_id / MASK_SIZE; // to access the right mask
            mask_t node_bit = 1 << (node_id % MASK_SIZE); // to access the right bit in the mask 
            s::atomic_ref<mask_t, s::memory_order::relaxed, s::memory_scope::work_group, s::access::address_space::local_space> next_ar{next[node_mask_offet]};

            if (parents_acc[node_offset + node_id] == -1) {
              for (int i = offsets_acc[node_offset + node_id]; i < offsets_acc[node_offset + node_id + 1]; i++) {
                nodeid_t neighbor = edges_acc[i];
                int neighbor_mask_offset = neighbor / MASK_SIZE;
                mask_t neighbor_bit = 1 << (neighbor % MASK_SIZE);
                if (frontier[neighbor_mask_offset] & neighbor_bit) {
                  parents_acc[node_offset + node_id] = neighbor;
                  next_ar |= node_bit;
                  break;
                }
              }
            }
          }
          
          if (loc_id == 0) running_ar.store(0);
          item.barrier(s::access::fence_space::local_space);
          if (loc_id < NUM_MASKS) {
            running_ar += next[loc_id];
          }
          item.barrier(s::access::fence_space::local_space);
        }
      }); 
    });
    events.push_back(e);
    e.wait_and_throw();
  }

  /**
   * @brief This method performs the BFS on multiple graphs using a bottom-up approach.
   * 
   * @param queue The SYCL queue to submit the kernel to.
   * @param data The vectorized graph data.
   * @param sources The vector of source nodes.
   * @param events The vector of events to be updated with the new event.
   * @param wg_size The size of the work-group to be used in the kernel.
   */
  void operator()(s::queue &queue, SYCL_VectorizedGraphData &data, const std::vector<nodeid_t> &sources, std::vector<s::event> &events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE)
  {
    s::range<1> global{wg_size * (data.data.size())}; // each workgroup will process a graph
    s::range<1> local{wg_size};

    s::buffer<nodeid_t, 1> sources_buf{sources.data(), s::range<1>{sources.size()}};

    auto e = queue.submit([&](s::handler &cgh) {
      size_t n_nodes [MAX_PARALLEL_GRAPHS];

      s::accessor<size_t, 1, s::access::mode::read> offsets_acc[MAX_PARALLEL_GRAPHS];
      s::accessor<nodeid_t, 1, s::access::mode::read> edges_acc[MAX_PARALLEL_GRAPHS];
      s::accessor<nodeid_t, 1, s::access::mode::read_write> parents_acc[MAX_PARALLEL_GRAPHS];
      s::accessor<nodeid_t, 1, s::access::mode::read> sources_acc{sources_buf, cgh, s::read_only};

      for (int i = 0; i < data.data.size(); i++) {
        offsets_acc[i] = data.offsets[i].get_access<s::access::mode::read>(cgh);
        edges_acc[i] = data.edges[i].get_access<s::access::mode::read>(cgh);
        parents_acc[i] = data.parents[i].get_access<s::access::mode::read_write>(cgh);
        n_nodes[i] = data.data[i].num_nodes;
      }

      typedef uint64_t mask_t;
      const unsigned MASK_SIZE = 64; // the size of the mask according to the type of mask_t
      const size_t MAX_NODES = *std::max_element(&n_nodes[0], &n_nodes[data.data.size() - 1]); // get the max number of nodes in graph
      const unsigned NUM_MASKS = MAX_NODES / MASK_SIZE + 1; // the number of masks needed to represent all nodes
      s::local_accessor<mask_t, 1> frontier{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> next{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> running{s::range<1>{1}, cgh};

      cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
        s::atomic_ref<mask_t, s::memory_order::relaxed, s::memory_scope::work_group> running_ar{running[0]};
        auto grp_id = item.get_group_linear_id();
        auto loc_id = item.get_local_id(0);
        auto local_size = item.get_local_range(0);

        auto offsets = offsets_acc[grp_id];
        auto edges = edges_acc[grp_id];
        auto parents = parents_acc[grp_id];
        auto node_count = n_nodes[grp_id];

        if (loc_id == 0) {
          running_ar.store(1);
          auto source = sources_acc[grp_id];
          int source_offset = source / MASK_SIZE;
          mask_t source_bit = 1 << (source % MASK_SIZE);
          frontier[source_offset] = next[source_offset] = source_bit;
        }

        item.barrier(s::access::fence_space::local_space);
        while (running_ar.load()) {
          if (loc_id < NUM_MASKS) {
            frontier[loc_id] = next[loc_id];
            next[loc_id] = 0;
          }
          item.barrier(s::access::fence_space::local_space);

          for (nodeid_t node_id = loc_id; node_id < node_count; node_id += local_size) {
            int node_mask_offet = node_id / MASK_SIZE; // to access the right mask
            mask_t node_bit = 1 << (node_id % MASK_SIZE); // to access the right bit in the mask 
            s::atomic_ref<mask_t, s::memory_order::relaxed, s::memory_scope::work_group> next_ar{next[node_mask_offet]};
            if (parents[node_id] == -1) {
              for (int i = offsets[node_id]; i < offsets[node_id + 1]; i++) {
                nodeid_t neighbor = edges[i];
                int neighbor_mask_offset = neighbor / MASK_SIZE;
                mask_t neighbor_bit = 1 << (neighbor % MASK_SIZE);
                if (frontier[neighbor_mask_offset] & neighbor_bit) {
                  parents[node_id] = neighbor;
                  next_ar |= node_bit;
                  break;
                }
              }
            }
          }
          
          running[0] = 0;
          item.barrier(s::access::fence_space::local_space);
          if (loc_id < NUM_MASKS) {
            running_ar += next[loc_id];
          }
          item.barrier(s::access::fence_space::local_space);
        }
      }); });
    events.push_back(e);
    e.wait_and_throw();
  }
};

#endif
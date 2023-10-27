#ifndef __BOTTOM_UP_OP_HPP__
#define __BOTTOM_UP_OP_HPP__

#include "impl/mul_bfs.hpp"

// TODO fix: Weird behaviour when executing with multiple graphs
// TODO: set the BFS source dynamically

namespace s = sycl;

template <size_t sg_size = 16>
class BottomUpMBFSOperator : public MultiBFSOperator {
  void operator() (s::queue& queue, SYCL_CompressedGraphData& data, std::vector<s::event>& events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE) {
    s::range<1> global{wg_size * (data.host_data.graphs_offsets.size() - 1)}; // each workgroup will process a graph
    s::range<1> local{wg_size};

    auto e = queue.submit([&](s::handler& cgh) {
      s::accessor offsets_acc{data.edges_offsets, cgh, s::read_only};
      s::accessor edges_acc{data.edges, cgh, s::read_only};
      s::accessor distances_acc{data.distances, cgh, s::read_write};
      s::accessor parents_acc{data.parents, cgh, s::read_write};
      s::accessor graphs_offsets_acc{data.graphs_offests, cgh, s::read_only};
      s::accessor nodes_offsets_acc{data.nodes_offsets, cgh, s::read_only};
      s::accessor nodes_count_acc{data.nodes_count, cgh, s::read_only};

      typedef uint64_t mask_t;
      const unsigned MASK_SIZE = 64; // the size of the mask according to the type of mask_t
      const size_t MAX_NODES = *std::max_element(data.host_data.nodes_count.begin(), data.host_data.nodes_count.end()); // get the max number of nodes in graph
      const unsigned NUM_MASKS = MAX_NODES / MASK_SIZE + 1; // the number of masks needed to represent all nodes
      s::local_accessor<mask_t, 1> frontier{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> next{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<int, 1> running{s::range<1>{1}, cgh};

      s::stream os {1024, 256, cgh}; // TODO REMOVE


      cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
        s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::work_group> running_ar{running[0]};
        auto grp_id = item.get_group_linear_id();
        auto loc_id = item.get_local_id(0);
        auto edge_offset = graphs_offsets_acc[grp_id];
        auto node_offset = nodes_offsets_acc[grp_id];
        auto node_count = nodes_count_acc[grp_id];
        auto local_size = item.get_local_range(0);

        if (loc_id == 0) {
          running_ar.store(1);
          frontier[0] = next[0] = 1;
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
            if (parents_acc[node_offset + node_id] == -1) {
              for (int i = offsets_acc[node_offset + node_id]; i < offsets_acc[node_offset + node_id + 1]; i++) {
                nodeid_t neighbor = edges_acc[i];
                int neighbor_mask_offset = neighbor / MASK_SIZE;
                mask_t neighbor_bit = 1 << (neighbor % MASK_SIZE);
                if (frontier[neighbor_mask_offset] & neighbor_bit) {
                  parents_acc[node_offset + node_id] = neighbor;
                  distances_acc[node_offset + node_id] = distances_acc[node_offset + neighbor] + 1;
                  next_ar |= node_bit;
                  break;
                }
              }
            }
          }
          
          running[0] = 0;
          item.barrier(s::access::fence_space::local_space);
          if (loc_id < NUM_MASKS) {
            running_ar.store(running_ar.load() || next[loc_id], s::memory_order::acq_rel);
          }
          item.barrier(s::access::fence_space::local_space);
        }
      });
    });
    events.push_back(e);
    e.wait_and_throw();
  }

  void operator() (s::queue& queue, SYCL_VectorizedGraphData& data, std::vector<s::event>& events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE) {
    s::range<1> global{wg_size * (data.data.size())}; // each workgroup will process a graph
    s::range<1> local{wg_size};

    auto e = queue.submit([&](s::handler& cgh) {

      size_t n_nodes [MAX_PARALLEL_GRAPHS];

      s::accessor<size_t, 1, s::access::mode::read> offsets_acc[MAX_PARALLEL_GRAPHS];
      s::accessor<nodeid_t, 1, s::access::mode::read> edges_acc[MAX_PARALLEL_GRAPHS];
      s::accessor<nodeid_t, 1, s::access::mode::read_write> parents_acc[MAX_PARALLEL_GRAPHS];
      s::accessor<distance_t, 1, s::access::mode::read_write> distances_acc[MAX_PARALLEL_GRAPHS];

      for (int i = 0; i < data.data.size(); i++) {
        offsets_acc[i] = data.offsets[i].get_access<s::access::mode::read>(cgh);
        edges_acc[i] = data.edges[i].get_access<s::access::mode::read>(cgh);
        parents_acc[i] = data.parents[i].get_access<s::access::mode::read_write>(cgh);
        distances_acc[i] = data.distances[i].get_access<s::access::mode::read_write>(cgh);
        n_nodes[i] = data.data[i].num_nodes;
      }

      typedef uint64_t mask_t;
      const unsigned MASK_SIZE = 64; // the size of the mask according to the type of mask_t
      const size_t MAX_NODES = *std::max_element(&n_nodes[0], &n_nodes[data.data.size() - 1]); // get the max number of nodes in graph
      const unsigned NUM_MASKS = MAX_NODES / MASK_SIZE + 1; // the number of masks needed to represent all nodes
      s::local_accessor<mask_t, 1> frontier{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<mask_t, 1> next{s::range<1>{NUM_MASKS}, cgh};
      s::local_accessor<int, 1> running{s::range<1>{1}, cgh};

      cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
        s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::work_group> running_ar{running[0]};
        auto grp_id = item.get_group_linear_id();
        auto loc_id = item.get_local_id(0);
        auto local_size = item.get_local_range(0);

        auto offsets = offsets_acc[grp_id];
        auto edges = edges_acc[grp_id];
        auto parents = parents_acc[grp_id];
        auto distances = distances_acc[grp_id];
        auto node_count = n_nodes[grp_id];

        if (loc_id == 0) {
          running_ar.store(1);
          frontier[0] = next[0] = 1;
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
                  distances[node_id] = distances[neighbor] + 1;
                  next_ar |= node_bit;
                  break;
                }
              }
            }
          }
          
          running[0] = 0;
          item.barrier(s::access::fence_space::local_space);
          if (loc_id < NUM_MASKS) {
            running_ar.store(running_ar.load() || next[loc_id], s::memory_order::acq_rel);
          }
          item.barrier(s::access::fence_space::local_space);
        }
      });
    });
    events.push_back(e);
    e.wait_and_throw();
  } 
};

#endif
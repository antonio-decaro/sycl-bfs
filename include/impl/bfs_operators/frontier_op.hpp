/**
 * @file frontier_op.hpp
 * @brief This file contains the implementation of the BFS operators that use a frontier-based approach.
 */
#ifndef __FRONTIER_OP_HPP__
#define __FRONTIER_OP_HPP__

#include "impl/mul_bfs.hpp"
#include "impl/simpl_bfs.hpp"
#include "kernel_sizes.hpp"

/**
 * @brief This class implements the BFS operator that uses a frontier-based approach for multiple graphs.
 * 
 * @tparam sg_size The size of the sub-group to be used in the kernel.
 */
template<size_t sg_size = 16>
class FrontierMBFSOperator : public MultiBFSOperator {
public:
  FrontierMBFSOperator() = default;

  /**
   * @brief This method performs the BFS on multiple graphs using a frontier-based approach.
   * 
   * @param queue The SYCL queue to submit the kernel to.
   * @param data The compressed graph data.
   * @param sources The vector of source nodes.
   * @param events The vector of events to be updated with the new event.
   * @param wg_size The size of the work-group to be used in the kernel.
   */
  void operator() (s::queue& queue, SYCL_CompressedGraphData& data, const std::vector<nodeid_t> &sources, std::vector<s::event>& events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE) {
    s::range<1> global{wg_size * (data.host_data.graphs_offsets.size() - 1)}; // each workgroup will process a graph
    s::range<1> local{wg_size};

    s::buffer<nodeid_t, 1> sources_buf{sources.data(), s::range<1>{sources.size()}};

    auto e = queue.submit([&](s::handler& cgh) {
      s::accessor offsets_acc{data.edges_offsets, cgh, s::read_only};
      s::accessor edges_acc{data.edges, cgh, s::read_only};
      s::accessor parents_acc{data.parents, cgh, s::read_write, s::no_init};
      s::accessor graphs_offsets_acc{data.graphs_offests, cgh, s::read_only};
      s::accessor nodes_offsets_acc{data.nodes_offsets, cgh, s::read_only};
      s::accessor nodes_count_acc{data.nodes_count, cgh, s::read_only};
      s::accessor sources_acc{sources_buf, cgh, s::read_only};

      typedef int fsize_t;
      s::local_accessor<fsize_t, 1> frontier{s::range<1>{wg_size}, cgh};
      s::local_accessor<size_t, 1> fsize_curr{s::range<1>{1}, cgh};
      s::local_accessor<size_t, 1> fsize_prev{s::range<1>{1}, cgh};

      cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) [[intel::reqd_sub_group_size(sg_size)]] {
          s::atomic_ref<size_t, s::memory_order::acq_rel, s::memory_scope::work_group> fsize_curr_ar{fsize_curr[0]};
          auto grp_id = item.get_group_linear_id();
          auto loc_id = item.get_local_id(0);
          auto edge_offset = graphs_offsets_acc[grp_id];
          auto node_offset = nodes_offsets_acc[grp_id];
          auto node_count = nodes_count_acc[grp_id];
          auto local_size = item.get_local_range(0);

          // init frontier
          if (loc_id == 0) {
            frontier[0] = sources_acc[grp_id];
            fsize_prev[0] = 1;
          }
          
          item.barrier(s::access::fence_space::local_space);
          while (fsize_prev[0] > 0) {
              if (loc_id < fsize_prev[0]) {
                  nodeid_t node = frontier[loc_id];
                  for (int i = offsets_acc[node_offset + node]; i < offsets_acc[node_offset + node + 1]; i++) {
                      nodeid_t neighbor = edges_acc[i];
                      if (parents_acc[node_offset + neighbor] == -1) {
                          parents_acc[node_offset + neighbor] = node;
                          auto pos = fsize_curr_ar.fetch_add(1);
                          frontier[pos] = neighbor;
                      }
                  }
              }
              item.barrier(s::access::fence_space::local_space);
              if (loc_id == 0) {
                  fsize_prev[0] = fsize_curr[0];
                  fsize_curr[0] = 0;
              }
              item.barrier(s::access::fence_space::local_space);
          }
      });
    });
    events.push_back(e);
    e.wait_and_throw();
  }

  /**
   * @brief This method performs the BFS on multiple graphs using a frontier-based approach.
   * 
   * @param queue The SYCL queue to submit the kernel to.
   * @param data The vectorized graph data.
   * @param sources The vector of source nodes.
   * @param events The vector of events to be updated with the new event.
   * @param wg_size The size of the work-group to be used in the kernel.
   */
  void operator() (s::queue& queue, SYCL_VectorizedGraphData& data, const std::vector<nodeid_t> &sources, std::vector<s::event>& events, const size_t wg_size = DEFAULT_WORK_GROUP_SIZE) {
    s::range<1> global{DEFAULT_WORK_GROUP_SIZE * (data.data.size())}; // each workgroup will process a graph
    s::range<1> local{DEFAULT_WORK_GROUP_SIZE};

    s::buffer<nodeid_t, 1> sources_buf{sources.data(), s::range<1>{sources.size()}};

    auto e = queue.submit([&](s::handler& cgh) {
      constexpr size_t ACC_SIZE = 8;

      size_t n_nodes [ACC_SIZE];
      s::accessor<size_t, 1, s::access::mode::read> offsets_acc[ACC_SIZE];
      s::accessor<nodeid_t, 1, s::access::mode::read> edges_acc[ACC_SIZE];
      s::accessor<nodeid_t, 1, s::access::mode::read_write> parents_acc[ACC_SIZE];
      s::accessor<nodeid_t, 1, s::access::mode::read> sources_acc{sources_buf, cgh, s::read_only};

      for (int i = 0; i < data.data.size(); i++) {
        offsets_acc[i] = data.offsets[i].get_access<s::access::mode::read>(cgh);
        edges_acc[i] = data.edges[i].get_access<s::access::mode::read>(cgh);
        parents_acc[i] = data.parents[i].get_access<s::access::mode::read_write>(cgh);
        n_nodes[i] = data.data[i].num_nodes;
      }

      typedef int fsize_t;
      s::local_accessor<fsize_t, 1> frontier{s::range<1>{DEFAULT_WORK_GROUP_SIZE}, cgh};
      s::local_accessor<size_t, 1> fsize_curr{s::range<1>{1}, cgh};
      s::local_accessor<size_t, 1> fsize_prev{s::range<1>{1}, cgh};

      cgh.parallel_for(s::nd_range<1>{global, local}, [=](s::nd_item<1> item) {
        s::atomic_ref<size_t, s::memory_order::acq_rel, s::memory_scope::work_group> fsize_curr_ar{fsize_curr[0]};
        auto grp_id = item.get_group_linear_id();
        auto loc_id = item.get_local_id(0);

        auto parents = parents_acc[grp_id];
        auto offsets = offsets_acc[grp_id];
        auto edges = edges_acc[grp_id];
        auto nodes_count = n_nodes[grp_id];
        auto local_size = item.get_local_range(0);

        for (int i = loc_id; i < nodes_count; i += local_size) {
          if (parents[i] == sources_acc[grp_id]) {
            frontier[0] = i;
            break;
          }
        }

        if (loc_id == 0) {
          fsize_prev[0] = 1;
        }
        
        item.barrier(s::access::fence_space::local_space);
        while (fsize_prev[0] > 0) {
          if (loc_id < fsize_prev[0]) {
            nodeid_t node = frontier[loc_id];
            for (int i = offsets[node]; i < offsets[node + 1]; i++) {
              nodeid_t neighbor = edges[i];
              if (parents[neighbor] == -1) {
                parents[neighbor] = node;
                auto pos = fsize_curr_ar.fetch_add(1);
                frontier[pos] = neighbor;
              }
            }
          }
          item.barrier(s::access::fence_space::local_space);
          if (loc_id == 0) {
            fsize_prev[0] = fsize_curr[0];
            fsize_curr[0] = 0;
          }
          item.barrier(s::access::fence_space::local_space);
        }
      });
    });
    events.push_back(e);
    e.wait_and_throw();
  } 
};

/**
 * @brief This class implements the BFS operator that uses a frontier-based approach for a single graph.
 * 
 * @tparam sg_size The size of the sub-group to be used in the kernel.
 */
template <size_t sg_size = 16>
class FrontierBFSOperator : public SingleBFSOperator {
public:
  /**
   * @brief This method performs the BFS on a single graph using a frontier-based approach.
   * 
   * @param queue The SYCL queue to submit the kernel to.
   * @param data The simple graph data.
   * @param events The vector of events to be updated with the new event.
   */
  void operator() (sycl::queue& queue, SYCL_SimpleGraphData& data, std::vector<sycl::event>& events) {
    int* frontier = s::malloc_device<int>(data.num_nodes, queue);
    int* frontier_size = s::malloc_shared<int>(1, queue);
    int* old_frontier_size = s::malloc_shared<int>(1, queue);
    queue.fill(frontier, 0, data.num_nodes).wait(); // init the frontier with the the node 0
    queue.fill(frontier_size, 0, 1).wait();
    queue.fill(old_frontier_size, 1, 1).wait();

    int level = 0;
    while (*old_frontier_size) {
      auto e = queue.submit([&](s::handler& h) {
        s::accessor offsets_acc(data.edges_offsets, h, s::read_only);
        s::accessor edges_acc(data.edges, h, s::read_only);
        s::accessor parents_acc(data.parents, h, s::read_write);

        size_t size = *old_frontier_size;
        h.parallel_for(s::range<1>{size}, [=](s::id<1> idx) [[intel::reqd_sub_group_size(sg_size)]] {
          s::atomic_ref<int, s::memory_order::relaxed, s::memory_scope::device> frontier_size_ref(*frontier_size);
          int node = frontier[idx[0]];
          
          for (int i = offsets_acc[node]; i < offsets_acc[node + 1]; i++) {
            int neighbor = edges_acc[i];
            if (parents_acc[neighbor] == -1) {
              int pos = frontier_size_ref.fetch_add(1);
              parents_acc[neighbor] = node;
              frontier[pos] = neighbor;
            }
          }

        });
      });
      events.push_back(e);
      e.wait();
      *old_frontier_size = *frontier_size;
      *frontier_size = 0;
    }

    s::free(frontier, queue);
    s::free(frontier_size, queue);
    s::free(old_frontier_size, queue);
  }
};

#endif
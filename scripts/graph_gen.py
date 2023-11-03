#!/usr/bin/env python3

import networkx as nx
import sys

def generate_random_graph(num_nodes: int, edges_p: float):
    """
    Generate a random graph with num_nodes nodes and edges_p probability of
    having an edge between two nodes.
    """
    return nx.fast_gnp_random_graph(num_nodes, edges_p, directed=False)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python graph_gen.py <num_nodes> <edges_p>')
        sys.exit(1)

    num_nodes, edges_p = map(int, sys.argv[1:])
    edges_p = edges_p / 100


    graph: nx.Graph = generate_random_graph(num_nodes, edges_p)
    
    # remove non-connected nodes
    graph.remove_nodes_from(list(nx.isolates(graph)))
    # remove self loops
    graph.remove_edges_from(nx.selfloop_edges(graph))

    print(graph.number_of_nodes())
    for edge in graph.edges:
        print(edge[0], edge[1])
        print(edge[1], edge[0])

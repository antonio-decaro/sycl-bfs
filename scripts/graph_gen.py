#!/usr/bin/env python3
import sys
import random
import networkx as nx


labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
use_wildcard = False

if __name__ == '__main__':
  if len(sys.argv) < 5:
    print('Usage: python graph_gen.py <n> <p> <out_data> <out_query> [use_wildcard]')
    exit(1)
  if len(sys.argv) == 6:
    use_wildcard = sys.argv[5] == 'true'
  

  n = int(sys.argv[1])
  p = float(int(sys.argv[2]) / 100.0)

  G: nx.Graph = nx.fast_gnp_random_graph(n, p)
  G.remove_nodes_from(list(nx.isolates(G)))
  # reset node index
  G = nx.convert_node_labels_to_integers(G)
  G = nx.DiGraph(G)
  n = len(G.nodes)

  for i in range(n):
    G.nodes[i]['label'] = random.choice(labels)

  with open(sys.argv[3], 'w') as f:
    print(n, len(G.edges), file=f)
    for i in range(len(G.nodes)):
      print(G.nodes[i]['label'], file=f)
    
    for edge in G.edges:
      print(edge[0], edge[1], file=f)

  # generate random subgraph of G
  subgraph = nx.Graph(G.subgraph(random.sample(G.nodes, random.randint(3, n//1.5))))
  # remove isolated nodes
  subgraph.remove_nodes_from(list(nx.isolates(subgraph)))
  # only one component
  to_remove = [i for i in subgraph if i not in max(list(nx.connected_components(subgraph)), key=len)]
  subgraph.remove_nodes_from(to_remove)
  # reset indices
  subgraph = nx.convert_node_labels_to_integers(subgraph)
  subgraph = nx.DiGraph(subgraph)
  if use_wildcard:
    wildcard_labels = random.sample(subgraph.nodes, random.randint(0, len(subgraph.nodes)//3))
    for node in wildcard_labels:
      subgraph.nodes[node]['label'] = 0

  with open(sys.argv[4], 'w') as f:
    print(len(subgraph.nodes), len(subgraph.edges), file=f)
    for i in subgraph.nodes:
      print(subgraph.nodes[i]['label'], file=f)
    for edge in subgraph.edges:
      print(edge[0], edge[1], file=f)
  
import sys
from collections import defaultdict



class Tree:
  def __init__(self) -> None:
    self._neighborhood = defaultdict(lambda: [])
    self.root = None
    self.nodes = set()
    self.valid = True
  def __getitem__(self, key):
    return self._neighborhood[key]
  def __setitem__(self, key, value):
    self._neighborhood[key] = value
  def add(self, node):
    self.nodes.add(node)
  def __contains__(self, node):
    return node in self.nodes

  def check_loop(self):
    loop = set()
    visited = set()
    stack = []
    stack.append(self.root)
    while stack:
      node = stack.pop()
      if node in visited:
        loop.add(node)
      else:
        visited.add(node)
        for n in self[node]:
          stack.append(n)
    return loop

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print('Usage: python check_bfs.py <graph_output_file>')
    exit(1)

  with open(sys.argv[1]) as f:
    lines = f.readlines()

  starts = []

  for i, line in enumerate(lines):
    lines[i] = line.strip().replace(' ', '').replace('\t', '')
    if '[!!!]' in line:
      starts.append(i)

  # delete the firsts info
  graphs_raw = []

  for i in range(len(starts) - 1):
    graphs_raw.append(lines[starts[i]:starts[i + 1]][1:])

  bfs_trees = []

  for num_graph, graph_raw in enumerate(graphs_raw):
    tree = Tree()
    for line in graph_raw:
      line: str
      n, parent, dist = line.split('|')
      n = int(n.split(':')[1])
      parent = int(parent.split(':')[1])
      dist = int(dist.split(':')[1])
      if n == parent:
        tree.root = n
        tree.add(n)
      elif parent == -1:
        tree.valid = False
      else:
        tree[parent].append(n)
        tree.add(n)
        tree.add(parent)
    bfs_trees.append(tree)

  for i, tree in enumerate(bfs_trees):
    if tree.valid:
      print(f'All the nodes in graph {i} have been visited :)')
      if len(l:= tree.check_loop()):
        print(f'[!] Loop detected in graph {i}:', l)
        exit(1)
      else:
        print(f"No loop detected in graph {i} :)")
    else:
      print(f'[!] Not all the nodes in graph {i} have been visited :(')
    print()
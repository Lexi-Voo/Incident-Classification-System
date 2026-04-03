from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

"""
Breadth-first search (level-order expansion) over a directed graph.

Args:
    edges: mapping node -> list of (neighbor, cost) pairs
    start: start node id
    goals: iterable of goal node ids

Returns:
    (goal_node_or_None, nodes_expanded, path_list)

Behavior notes:
    - Expands nodes one level at a time (FIFO queue).
    - Uses parent pointers to reconstruct the path to the found goal.
    - Neighbors are visited in sorted order for deterministic results.
    - nodes_expanded counts how many nodes were dequeued/expanded.
"""
def bfs_search(edges: Dict[int, List[Tuple[int, float]]], start: int, goals: Iterable[int]) -> Tuple[Optional[int], int, List[int]]:
   
    goal_set = set(goals)

    queue = deque([start])
    parent: Dict[int, Optional[int]] = {start: None}
    visited = {start}
    num_nodes = 0

    while queue:
        node = queue.popleft()
        num_nodes += 1

        if node in goal_set:
            # Reconstruct path using parent pointers
            path: List[int] = []
            cur = node
            while cur is not None:
                path.append(cur)
                cur = parent.get(cur)
            path.reverse()
            return node, num_nodes, path

        # Enqueue neighbors in sorted order for deterministic traversal
        for neighbor, _ in sorted(edges.get(node, []), key=lambda x: x[0]):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    return None, num_nodes, []

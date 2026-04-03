import heapq, math, itertools
from typing import Dict, List, Tuple, Set, Optional

def gbfs_search(nodes: Dict[int, Tuple[float, float]],
                edges: Dict[int, List[Tuple[int, float]]],
                origin: int,
                destinations: List[int]) -> Tuple[Optional[int], int, List[int]]:
    """
    Greedy Best-First Search (GBFS)
    Expands the node with the lowest heuristic (Euclidean distance to goal).
    """
    def heuristic(nid: int) -> float:
        x1, y1 = nodes[nid]
        return min(math.hypot(x1 - nodes[g][0], y1 - nodes[g][1]) for g in destinations)

    counter = itertools.count()
    pq = []
    parent = {origin: None}
    nodes_created = 1
    heapq.heappush(pq, (heuristic(origin), next(counter), origin))
    visited: Set[int] = set()

    while pq:
        _, _, current = heapq.heappop(pq)
        if current in destinations:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[-1], nodes_created, list(reversed(path))

        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in sorted(edges.get(current, []), key=lambda x: x[0]):
            if neighbor not in visited:
                parent[neighbor] = current
                heapq.heappush(pq, (heuristic(neighbor), next(counter), neighbor))
                nodes_created += 1

    return None, nodes_created, []

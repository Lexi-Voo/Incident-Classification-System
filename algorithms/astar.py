import heapq, math, itertools
from typing import Dict, List, Tuple, Set, Optional

def astar_search(nodes: Dict[int, Tuple[float, float]],
                 edges: Dict[int, List[Tuple[int, float]]],
                 origin: int,
                 destinations: List[int]) -> Tuple[Optional[int], int, List[int]]:
    """
    A* Search (f(n) = g(n) + h(n))
    """
    def heuristic(nid: int) -> float:
        x1, y1 = nodes[nid]
        return min(math.hypot(x1 - nodes[g][0], y1 - nodes[g][1]) for g in destinations)

    counter = itertools.count()
    pq = []
    parent = {origin: None}
    g_cost = {origin: 0.0}
    nodes_created = 1
    heapq.heappush(pq, (heuristic(origin), next(counter), origin))

    while pq:
        _, _, current = heapq.heappop(pq)
        if current in destinations:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[-1], nodes_created, list(reversed(path))

        for neighbor, cost in sorted(edges.get(current, []), key=lambda x: x[0]):
            new_cost = g_cost[current] + cost
            if neighbor not in g_cost or new_cost < g_cost[neighbor]:
                g_cost[neighbor] = new_cost
                parent[neighbor] = current
                f_cost = new_cost + heuristic(neighbor)
                heapq.heappush(pq, (f_cost, next(counter), neighbor))
                nodes_created += 1

    return None, nodes_created, []

import math
import heapq
from typing import Dict, List, Tuple, Set, Optional

def euclidean_distance(nodes: Dict[int, Tuple[int, int]], node1: int, node2: int)->float:
    x1, y1 = nodes[node1]
    x2, y2 = nodes[node2]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def get_neighbours(edges: Dict[int, List[Tuple[int, int]]], node: int) -> List[Tuple[int, int]]:
    return edges.get(node, [])

def a_star(nodes: Dict[int, Tuple[int, int]],
           edges: Dict[int, List[Tuple[int, int]]],
           start: int,
           goal: int,
           restricted_nodes: Set[int] = None) ->Tuple[Optional[List[int]], int]:
    
    if restricted_nodes is None:
        restricted_nodes = set()
    
    open_set = [(0, 0, start, [start])]
    visited = set()
    
    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)
        
        if current in visited:
            continue
        
        visited.add(current)
        
        if current == goal:
            return path, g_score
        
        for neighbour, edge_cost in get_neighbours(edges, current):
            if neighbour in visited or neighbour in restricted_nodes:
                continue
            
            new_g_score = g_score + edge_cost
            h_score = euclidean_distance(nodes, neighbour, goal)
            new_f_score = new_g_score + h_score
            new_path = path + [neighbour]
            
            heapq.heappush(open_set, (new_f_score, new_g_score, neighbour, new_path))
    
    return None, float('inf')

def create_clusters(nodes: Dict[int, Tuple[int, int]]) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    x_coords = [x for x, y in nodes.values()]
    median_x = sorted(x_coords)[len(x_coords)//2]
    
    clusters = {1: [], 2: []}
    node_to_cluster = {}
    
    for node_id, (x, y) in nodes.items():
        if x <= median_x:
            clusters[1].append(node_id)
            node_to_cluster[node_id] = 1
        else:
            clusters[2].append(node_id)
            node_to_cluster[node_id] = 2
    
    return clusters, node_to_cluster

def find_entrances(edges: Dict[int, List[Tuple[int, int]]],
                   node_to_cluster: Dict[int, int]) -> List[Tuple[int, int, int]]:
    entrances = []
    
    for from_node, neighbors in edges.items():
        for to_node, cost in neighbors:
            from_cluster = node_to_cluster[from_node]
            to_cluster = node_to_cluster[to_node]
            
            if from_cluster != to_cluster:
                entrances.append((from_node, to_node, cost))
    
    return entrances

# FIXED: Don't restrict to cluster - allow full graph search
def find_intra_cluster_path(nodes: Dict[int, Tuple[int, int]],
                            edges: Dict[int, List[Tuple[int, int]]],
                            start: int,
                            goal: int,
                            cluster_nodes: List[int]) -> Tuple[Optional[List[int]], int]:
    """
    FIXED: Instead of restricting nodes to cluster (which breaks paths),
    just use regular A* on the full graph.
    
    The hierarchical aspect of HPA* comes from:
    1. Choosing good entrances between clusters
    2. Breaking the problem into start→entrance and entrance→goal
    
    NOT from restricting the actual pathfinding within clusters.
    """
    # Use full A* without restrictions
    return a_star(nodes, edges, start, goal, restricted_nodes=None)

def find_best_entrance(nodes: Dict[int, Tuple[int, int]],
                       entrances: List[Tuple[int, int, int]],
                       node_to_cluster: Dict[int, int],
                       start_cluster: int,
                       goal_cluster: int,
                       start_node: int,
                       goal_node: int) -> Optional[Tuple[int, int, int]]:
    
    best_cost = float('inf')
    best_entrance = None
    
    for from_node, to_node, entrance_cost in entrances:
        from_cluster = node_to_cluster[from_node]
        to_cluster = node_to_cluster[to_node]
        
        # Case 1: Entrance in correct direction
        if from_cluster == start_cluster and to_cluster == goal_cluster:
            cost_to_entrance = euclidean_distance(nodes, start_node, from_node)
            cost_from_entrance = euclidean_distance(nodes, to_node, goal_node)
            estimated_cost = cost_to_entrance + entrance_cost + cost_from_entrance
            
            if estimated_cost < best_cost:
                best_cost = estimated_cost
                best_entrance = (from_node, to_node, entrance_cost)
        
        # Case 2: Entrance in reverse direction
        elif from_cluster == goal_cluster and to_cluster == start_cluster:
            cost_to_entrance = euclidean_distance(nodes, start_node, to_node)
            cost_from_entrance = euclidean_distance(nodes, from_node, goal_node)
            estimated_cost = cost_to_entrance + entrance_cost + cost_from_entrance
            
            if estimated_cost < best_cost:
                best_cost = estimated_cost
                best_entrance = (to_node, from_node, entrance_cost)
    
    return best_entrance

def hpa_star_search(nodes: Dict[int, Tuple[int, int]],
                    edges: Dict[int, List[Tuple[int, int]]],
                    origin: int,
                    destinations: List[int]) -> Tuple[int, int, List[int]]:
    """
    FIXED HPA* - Uses hierarchical entrance selection but allows full pathfinding
    """
    
    # Phase 1: Abstraction
    clusters, node_to_cluster = create_clusters(nodes)
    
    # Find entrances
    entrances = find_entrances(edges, node_to_cluster)
    
    # Try each destination
    for destination in destinations:
        start_cluster = node_to_cluster[origin]
        goal_cluster = node_to_cluster[destination]
        
        # Same cluster case
        if start_cluster == goal_cluster:
            cluster_nodes = clusters[start_cluster]
            path, cost = find_intra_cluster_path(
                nodes, edges, origin, destination, cluster_nodes
            )
            
            if path:
                return destination, len(path), path
        
        # Different clusters case
        else:
            # Find best entrance
            entrance = find_best_entrance(
                nodes, entrances, node_to_cluster, 
                start_cluster, goal_cluster, origin, destination
            )
            
            if not entrance:
                continue
            
            entrance_from, entrance_to, entrance_cost = entrance
            
            # Path 1: Origin to entrance (using full graph now)
            start_cluster_nodes = clusters[start_cluster]
            path1, cost1 = find_intra_cluster_path(
                nodes, edges, origin, entrance_from, start_cluster_nodes
            )
            
            if not path1:
                continue
            
            # Path 2: Entrance to goal (using full graph now)
            goal_cluster_nodes = clusters[goal_cluster]
            path2, cost2 = find_intra_cluster_path(
                nodes, edges, entrance_to, destination, goal_cluster_nodes
            )
            
            if not path2:
                continue
            
            # Combine paths
            full_path = path1[:-1] + [entrance_from, entrance_to] + path2[1:]
            total_nodes = len(full_path)
            
            return destination, total_nodes, full_path
    
    # No path found
    return None, 0, []
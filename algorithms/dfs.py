"""
Depth-first search (iterative) over a directed graph.

Args:
    edges: dict mapping node -> list of (neighbor, cost) pairs
    start: start node id
    goals: collection (iterable) of goal node ids

    Returns:
    (goal_node_or_None, nodes_expanded, path_list)

    Notes:
    - Converts goals to a set for fast membership checks.
    - Uses a stack of (node, path) and returns the first goal found.
    - Neighbors are visited in sorted order for deterministic behaviour."""
def dfs_search(edges, start, goals):
    
    # Normalize goals to a set for O(1) membership tests
    goal_set = set(goals)

    stack = [(start, [start])]
    visited = set()
    num_nodes = 0

    while stack:
        node, path = stack.pop()
        num_nodes += 1

        if node in goal_set:
            return node, num_nodes, path

        if node in visited:
            continue

        visited.add(node)

        # Iterate neighbors in sorted order to make results deterministic
        neighbors = edges.get(node, [])
        for neighbor, _ in sorted(neighbors, key=lambda x: x[0], reverse=True):
            if neighbor not in visited:
                # append a new path list (avoid mutating existing path)
                stack.append((neighbor, path + [neighbor]))
    # If we exhaust the stack and don't find any goal, return None
    return None, num_nodes, []
                                                                                                                                        

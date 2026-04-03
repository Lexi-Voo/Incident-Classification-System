"""
Fronter Novelty Search (FNS)
Expands nodes with the highest unexplored neighbors first, using below rule:
Selection Priority:
1. Highest Novelty Score
2. Smallest Node ID 
3. Smallest Insertion Index 
"""

def count_unexplored_neighbors(graph, node, explored, frontier_nodes):
    """
    Calculates the Novelty Score: The count of neighbors truly unvisited.
    A neighbor is 'unexplored' only if it is NOT in 'explored' AND NOT in 'frontier'.
    """
    if node not in graph:
        return 0
    
    # Exclude all nodes already visited or currently waiting for expansion
    visited_or_pending = explored.union(frontier_nodes)
    
    return sum(
        1 for neighbor, _ in graph.get(node, []) if neighbor not in visited_or_pending
    )

def fns_search(graph, start, goals):
    # Initialize frontier stores: [(current_node, path)]
    frontier = [(start, [start])]
    frontier_nodes = {start}       # Set for quick membership check
    explored = set()               # Set of explored nodes
    total_nodes = 0                # Number of nodes expanded (popped from frontier)

    # Find and Expand the Highest Priority Node
    while frontier:
        selection_keys = []
        for idx, (node, _) in enumerate(frontier):
            # Primary Priority: Calculate Novelty Score
            novelty = count_unexplored_neighbors(graph, node, explored, frontier_nodes)
            
            # The Max function selects the best:
            # - Max(Novelty)
            # - Max(-NodeID) => Smallest Node ID
            # - Max(-Index) => Smallest Index (FIFO)
            key = (novelty, -node, -idx)
            selection_keys.append((key, idx))

        # Select the node with the highest overall priority key
        best_key, best_index = None, -1
        for key, original_index in selection_keys:
            if best_key is None or key > best_key:
                best_key = key
                best_index = original_index
        
        # Pop and expand the selected node
        current_node, path = frontier.pop(best_index)
        frontier_nodes.remove(current_node)
        total_nodes += 1 # Increment expanded node count

        # Goal Test
        if current_node in goals:
            return current_node, total_nodes, path

        # Expand Node and Update Frontier
        explored.add(current_node)
        neighbors = graph.get(current_node, [])

        # Sort neighbors by Node ID (Assignment Requirement for deterministic expansion)
        sorted_neighbors = sorted(neighbors, key=lambda x: x[0])

        for neighbor, _ in sorted_neighbors:
            # Check for redundancy: only add truly unexplored nodes
            if neighbor not in explored and neighbor not in frontier_nodes:
                # Append ensures the chronological insertion order is maintained
                frontier.append((neighbor, path + [neighbor]))
                frontier_nodes.add(neighbor)

    # Returns None and the total nodes explored before exhausting the graph
    return None, total_nodes, []
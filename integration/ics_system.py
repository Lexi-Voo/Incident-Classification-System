import sys
import os
import copy

# Add path to algorithms folder
algorithms_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'algorithms')
if algorithms_path not in sys.path:
    sys.path.append(algorithms_path)

# Import from integration package (same folder)
from integration.accident_predictor import AccidentPredictor

# Import algorithms
from algorithms.astar import astar_search
from algorithms.bfs import bfs_search
from algorithms.dfs import dfs_search
from algorithms.gbfs import gbfs_search
from algorithms.hpa import hpa_star_search
from algorithms.fns import fns_search

# Import OSM helper (same folder)
from integration import osm_helper as osm


class ICSSystem:
    def __init__(self, model_path, graph_file='heritage_assignment_15.txt', osm_file='map.osm'):
        print("Initializing ICS System...")
        
        # 1. Load ML model
        try:
            self.predictor = AccidentPredictor(model_path)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            self.predictor = None

        # 2. Load Assignment Graph
        self.nodes, self.edges, self.cameras, self.way_to_edge = self.load_graph(graph_file)
        self.original_edges = copy.deepcopy(self.edges)
        self.current_edges_state = copy.deepcopy(self.edges)  # ✅ CRITICAL FIX
        self.ACCIDENT_MULTIPLIER = 3
        
        # 3. Load OSM Map
        self.edge_geometry = {} 
        self.osm_nodes = None
        self.osm_graph = None
        
        if os.path.exists(osm_file):
            try:
                self.osm_nodes, self.osm_graph = osm.load_osm_graph(osm_file)
                self.precompute_geometries()
            except Exception as e:
                print(f"⚠ Error loading OSM data: {e}")
        else:
            print(f"⚠ {osm_file} not found. Using straight lines.")

        print(f"✓ ICS System initialized successfully\n")

    def load_graph(self, filepath):
        """Load graph from text file"""
        nodes = {}
        edges = {}
        cameras = {}
        way_to_edge = {}
        self.node_names = {}
        current_section = None
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'): 
                        continue
                    
                    if line.startswith('['):
                        current_section = line.upper()
                        continue
                    
                    if current_section == '[NODES]':
                        parts = self.split_csv_allow_commas(line, 4)
                        nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
                        self.node_names[int(parts[0])] = parts[3].strip()
                    
                    elif current_section == '[WAYS]':
                        parts = self.split_csv_allow_commas(line, 6)
                        u, v = int(parts[1]), int(parts[2])
                        time = float(parts[5])
                        
                        if u not in edges: 
                            edges[u] = []
                        edges[u].append((v, time))
                        
                        way_to_edge[parts[0]] = {
                            'from': u, 'to': v, 'original_time': time,
                            'road_name': parts[3], 'highway_type': parts[4]
                        }
                    
                    elif current_section == '[CAMERAS]':
                        parts = line.split(',')
                        cameras[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""
                        
        except Exception as e:
            raise Exception(f"Error parsing graph file: {str(e)}")
        
        print(f"✓ Loaded graph: {len(nodes)} nodes, {sum(len(v) for v in edges.values())} edges")
        return nodes, edges, cameras, way_to_edge

    def split_csv_allow_commas(self, line, min_fields):
        """
        Split CSV line while allowing commas inside parentheses.
        Used for parsing the graph file where names may contain commas.
        """
        parts = []
        buf = []
        depth = 0
        
        for ch in line:
            if ch == '(': 
                depth += 1
            elif ch == ')': 
                depth = max(depth - 1, 0)
            elif ch == ',' and depth == 0:
                parts.append("".join(buf).strip())
                buf = []
                continue
            buf.append(ch)
        
        if buf: 
            parts.append("".join(buf).strip())
        
        return parts

    def precompute_geometries(self):
        """
        Pre-compute detailed road geometries for all edges in the assignment graph.
        This maps assignment edges to their actual road paths in the OSM network.
        """
        print("Pre-computing road geometries from OSM...")
        count = 0
        failed = []
        snapped_cache = {} 
        
        for u, neighbors in self.edges.items():
            # Snap node u to OSM network (with caching)
            if u not in snapped_cache:
                try:
                    nearest = osm.k_nearest_graph_nodes(
                        self.nodes[u][0], 
                        self.nodes[u][1], 
                        self.osm_nodes, 
                        self.osm_graph, 
                        k=1
                    )
                    snapped_cache[u] = nearest[0] if nearest else None
                except Exception as e:
                    print(f"⚠ Warning: Could not snap node {u}: {e}")
                    snapped_cache[u] = None
            
            u_osm = snapped_cache[u]
            if u_osm is None:
                continue

            for v, _ in neighbors:
                # Snap node v to OSM network (with caching)
                if v not in snapped_cache:
                    try:
                        nearest = osm.k_nearest_graph_nodes(
                            self.nodes[v][0], 
                            self.nodes[v][1], 
                            self.osm_nodes, 
                            self.osm_graph, 
                            k=1
                        )
                        snapped_cache[v] = nearest[0] if nearest else None
                    except Exception as e:
                        print(f"⚠ Warning: Could not snap node {v}: {e}")
                        snapped_cache[v] = None
                
                v_osm = snapped_cache[v]
                if v_osm is None:
                    continue
                
                # Get detailed path between snapped OSM nodes
                try:
                    path = osm.get_detailed_path(self.osm_graph, self.osm_nodes, u_osm, v_osm)
                    if path:
                        self.edge_geometry[(u, v)] = path
                        count += 1
                    else:
                        failed.append((u, v))
                except Exception as e:
                    print(f"⚠ Warning: Could not compute path for edge {u}→{v}: {e}")
                    failed.append((u, v))
        
        print(f"✓ Cached geometry for {count} edge segments")
        
        if failed:
            print(f"⚠ Warning: {len(failed)} edges could not be mapped to OSM:")
            for u, v in failed[:5]:  # Show first 5 failures
                print(f"  • Edge {u} → {v}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

    def get_path_geometry(self, node_path):
        """
        Constructs the complete geometry for a route by stitching together
        individual edge geometries from the OSM network.
        
        Args:
            node_path: List of node IDs representing the route
            
        Returns:
            List of (lat, lon) tuples representing waypoints along the route
        """
        if not node_path or len(node_path) < 2: 
            return []
        
        full_geo = []
        
        for i in range(len(node_path) - 1):
            u, v = node_path[i], node_path[i+1]
            seg = self.edge_geometry.get((u, v))
            
            if seg:
                # We have OSM geometry for this edge
                if i < len(node_path) - 2: 
                    # Not the last segment - exclude the last point to avoid duplicates
                    full_geo.extend(seg[:-1])
                else: 
                    # Last segment - include all points
                    full_geo.extend(seg)
            else:
                # No OSM geometry - fallback to straight line
                full_geo.append(self.nodes[u])
                if i == len(node_path) - 2: 
                    # Last segment - add destination
                    full_geo.append(self.nodes[v])
        
        return full_geo

    def find_k_routes(self, algo_name, algo_func, start, end, k=5):
        """
        Finds up to K unique paths for a specific algorithm.
        Strategy: Find path → Penalize edges in that path → Repeat.
        
        Args:
            algo_name: Name of the algorithm (e.g., 'A*', 'BFS')
            algo_func: Algorithm function to call
            start: Start node ID
            end: End node ID
            k: Number of alternative routes to find (default: 5)
            
        Returns:
            List of route dictionaries containing path, cost, and nodes expanded
        """
        # ✅ Validate inputs
        if start not in self.nodes:
            raise ValueError(f"Start node {start} not in graph")
        if end not in self.nodes:
            raise ValueError(f"End node {end} not in graph")
        
        found_routes = []
        seen_paths = set()  # ✅ Track unique paths
        
        # Create a working copy of edges that we can modify
        # Start with the current state (which may include accident penalties)
        temp_edges = copy.deepcopy(self.current_edges_state)
        
        for i in range(k):
            try:
                # 1. Run the algorithm
                if algo_name in ['BFS', 'DFS', 'FNS']:
                    # Uninformed search - explicit signature
                    result = algo_func(temp_edges, start, [end])
                else:
                    # Informed search - takes nodes parameter
                    result = algo_func(self.nodes, temp_edges, start, [end])
                
                # Unpack result (goal, nodes_expanded, path)
                _, nodes_exp, path = result
                
                if not path:
                    break  # No more paths possible
                
                # ✅ Check if this path is unique
                path_tuple = tuple(path)
                if path_tuple in seen_paths:
                    # Skip duplicate path, but still try to find more by penalizing edges
                    pass
                else:
                    # 2. Store this unique route
                    seen_paths.add(path_tuple)
                    
                    # Calculate cost based on CURRENT state (with accidents)
                    cost = self.calculate_path_cost(path, self.current_edges_state)
                    
                    found_routes.append({
                        'algo': algo_name,
                        'id': len(found_routes) + 1,  # Assign ID based on unique count
                        'path': path,
                        'cost': cost,
                        'nodes': nodes_exp
                    })
                
                # 3. Penalize edges in this path to force different route next time
                is_unweighted = algo_name in ['BFS', 'DFS', 'FNS']
                
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    
                    if u in temp_edges:
                        # Rebuild neighbor list with penalized/removed edge
                        new_neighbors = []
                        for nbr, c in temp_edges[u]:
                            if nbr == v:
                                if is_unweighted:
                                    # For unweighted search - REMOVE the edge completely
                                    continue
                                else:
                                    # For weighted search - make it very expensive
                                    new_neighbors.append((nbr, c * 1000.0))
                            else:
                                new_neighbors.append((nbr, c))
                        temp_edges[u] = new_neighbors
                        
            except Exception as e:
                print(f"⚠ Error in K-route loop for {algo_name}: {e}")
                break
                
        return found_routes

    def detect_accident_at_node(self, node_id, image_path):
        """
        Detect accident severity from an image using the ML model.
        
        Args:
            node_id: Node ID where accident is detected
            image_path: Path to the accident/damage image
            
        Returns:
            Dictionary with severity, class_name, confidence, etc.
        """
        if not self.predictor: 
            return {'severity': 0, 'class_name': 'Error', 'confidence': 0}
        
        return self.predictor.predict(image_path)

    def update_travel_times_for_node(self, node_accidents):
        """
        Updates edge weights based on accident severity at nodes.
        Creates a new edge dictionary with penalized travel times.
        
        Args:
            node_accidents: Dictionary mapping node_id to accident info
                           (must contain 'severity' key)
        
        Returns:
            Updated edges dictionary with accident penalties applied
        """
        # Start with original edges (no penalties)
        updated = copy.deepcopy(self.original_edges)
        
        for node_id, info in node_accidents.items():
            sev = info['severity']
            
            if sev == 0:  # No accident
                continue
            
            # Penalize edges FROM the accident node
            if node_id in updated:
                updated[node_id] = [
                    (n, t * sev * self.ACCIDENT_MULTIPLIER) 
                    for n, t in updated[node_id]
                ]
            
            # Penalize edges TO the accident node
            for from_n, neighbors in updated.items():
                if from_n == node_id:
                    continue  # Already handled above
                
                updated[from_n] = [
                    (n, t * sev * self.ACCIDENT_MULTIPLIER if n == node_id else t) 
                    for n, t in neighbors
                ]
        
        # Store this state for K-route finder to use as base
        self.current_edges_state = updated
        return updated

    def calculate_path_cost(self, path, edges):
        """
        Calculate the total cost of a path given an edge dictionary.
        
        Args:
            path: List of node IDs
            edges: Edge dictionary (node_id -> [(neighbor, cost), ...])
            
        Returns:
            Total cost (sum of edge weights)
        """
        cost = 0
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            for neighbor, c in edges.get(u, []):
                if neighbor == v:
                    cost += c
                    break
        return cost
        
    def get_node_name(self, node_id):
        """Get the human-readable name for a node"""
        return self.node_names.get(node_id, f"Node {node_id}")
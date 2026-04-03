import xml.etree.ElementTree as ET
import math
import heapq

def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two lat/lon points in km"""
    R = 6371.0
    la1, la2 = math.radians(lat1), math.radians(lat2)
    dla = la2 - la1
    dlo = math.radians(lon2 - lon1)
    a = math.sin(dla/2)**2 + math.cos(la1)*math.cos(la2)*math.sin(dlo/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

def load_osm_graph(osm_path):
    """Parses .osm file and returns nodes and adjacency list"""
    print(f"Loading OSM graph from {osm_path}...")
    tree = ET.parse(osm_path)
    root = tree.getroot()

    osm_nodes = {}
    for n in root.findall("node"):
        nid = n.attrib["id"]
        lat = float(n.attrib["lat"])
        lon = float(n.attrib["lon"])
        osm_nodes[nid] = (lat, lon)

    graph = {nid: [] for nid in osm_nodes}

    for w in root.findall("way"):
        nd_refs = [nd.attrib["ref"] for nd in w.findall("nd")]
        # Filter for valid highways if needed, or take all ways
        tags = {t.attrib.get("k"): t.attrib.get("v") for t in w.findall("tag")}
        if "highway" not in tags:
            continue

        for i in range(len(nd_refs) - 1):
            a, b = nd_refs[i], nd_refs[i+1]
            if a in osm_nodes and b in osm_nodes:
                lat1, lon1 = osm_nodes[a]
                lat2, lon2 = osm_nodes[b]
                dist = haversine_km(lat1, lon1, lat2, lon2)
                # Undirected graph for OSM roads
                graph[a].append((b, dist))
                graph[b].append((a, dist))

    # Clean up isolated nodes to save memory
    connected_nodes = {nid: coords for nid, coords in osm_nodes.items() if graph[nid]}
    
    print(f"✓ Loaded OSM: {len(connected_nodes)} nodes, {sum(len(v) for v in graph.values())} edges")
    return connected_nodes, graph

def k_nearest_graph_nodes(lat, lon, osm_nodes, graph, k=1):
    """Finds the k nearest OSM nodes to a given coordinate"""
    candidates = []
    for nid, (nlat, nlon) in osm_nodes.items():
        if nid not in graph:
            continue
        d_lat = lat - nlat
        d_lon = lon - nlon
        d2 = d_lat*d_lat + d_lon*d_lon
        candidates.append((d2, nid))
    candidates.sort(key=lambda x: x[0])
    return [nid for d2, nid in candidates[:k]]

def get_detailed_path(graph, osm_nodes, start_node, end_node):
    """Runs Dijkstra to find the shape of the road between two OSM nodes"""
    # If start and end are the same, return single point
    if start_node == end_node:
        return [osm_nodes[start_node]]
        
    dist = {start_node: 0.0}
    prev = {}
    pq = [(0.0, start_node)]
    visited = set()

    while pq:
        cur_d, cur = heapq.heappop(pq)
        if cur in visited:
            continue
        visited.add(cur)
        if cur == end_node:
            break

        for nbr, w in graph.get(cur, []):
            nd = cur_d + w
            if nbr not in dist or nd < dist[nbr]:
                dist[nbr] = nd
                prev[nbr] = cur
                heapq.heappush(pq, (nd, nbr))

    if end_node not in prev:
        return None # No path found

    # Reconstruct path
    path_coords = []
    curr = end_node
    while curr:
        path_coords.append(osm_nodes[curr])
        if curr == start_node:
            break
        curr = prev.get(curr)
    
    return list(reversed(path_coords))
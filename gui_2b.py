"""
Enhanced GUI for Assignment 2B - WITH IMAGE PREVIEW IN ACCIDENT LIST
Features:
- Individual route visibility checkboxes in Results Summary
- Color-coded routes with shades (darker = route #1, lighter = route #5)
- Algorithm-level controls disable/enable their route checkboxes
- Clean, intuitive route management
- IMAGE PREVIEW: Shows thumbnail of accident image next to accident type
"""

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sys
from PIL import Image, ImageTk  # Added for image handling

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'integration'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorithms'))

from integration.ics_system import ICSSystem
from algorithms.astar import astar_search
from algorithms.bfs import bfs_search
from algorithms.dfs import dfs_search
from algorithms.fns import fns_search
from algorithms.gbfs import gbfs_search
from algorithms.hpa import hpa_star_search

try:
    import tkintermapview
    HAS_MAP = True
    print("✓ TkinterMapView imported successfully")
except ImportError:
    HAS_MAP = False
    print("✗ TkinterMapView not found")

class EnhancedICS_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Incident Classification System - Assignment 2B (Group SkyNet)")
        self.root.state('zoomed')
        
        # Default files
        self.current_graph_file = 'testcases/heritage_assignment_15.txt'
        self.current_model_file = 'models/model2_mobilenetV2.h5'
        self.osm_file = 'map.osm'
        
        # State variables
        self.ics = None
        self.edge_accidents = {}  # Changed from node_accidents to edge_accidents
        self.all_routes = []
        self.route_changes = {}
        self.route_status = {}
        self.node_markers = {}
        self.route_paths = {}
        self.edge_paths = []
        self.accident_edge_paths = {}  # NEW: Store red highlighted accident edges
        self.route_time_marker = None  # NEW: Marker showing time for selected route
        
        # NEW: Store PIL Image references to prevent garbage collection
        self.accident_image_refs = {}
        self.selected_accident = None  # NEW: Track which accident is selected (now edge tuple)
        self.accident_frames = {}  # NEW: Store frame references for highlighting
        
        # NEW: Per-route visibility controls
        self.route_checkboxes = {}
        self.route_visibility = {}
        
        # NEW: Best routes mode
        self.viewing_mode = 'all_routes'  # 'all_routes' or 'best_routes'
        self.best_routes = []  # Store top 3 best routes with different paths
        self.best_route_checkboxes = {}
        self.best_route_visibility = {}
        self.best_routes_gray_paths = []  # Store gray paths for unselected best routes
        
        # Base colors for each algorithm
        self.algo_base_colors = {
            'A*': (0, 102, 204),
            'BFS': (0, 204, 102),
            'DFS': (204, 0, 204),
            'GBFS': (255, 153, 0),
            'HPA*': (152, 245, 249),  # Cyan color #98F5F9
            'FNS': (255, 204, 0)
        }
        
        # Create widgets first
        self.create_widgets()
        
        # Initialize system with defaults
        self.initialize_system()
    
    def get_route_color(self, algo, route_id, total_routes):
        """Generate color shades for routes of same algorithm"""
        base_r, base_g, base_b = self.algo_base_colors[algo]
        
        if total_routes == 1:
            return f'#{base_r:02x}{base_g:02x}{base_b:02x}'
        
        brightness = 0.6 + (route_id - 1) * (0.4 / (total_routes - 1))
        
        r = int(base_r * brightness)
        g = int(base_g * brightness)
        b = int(base_b * brightness)
        
        return f'#{r:02x}{g:02x}{b:02x}'
    
    def get_route_midpoint(self, path):
        """Calculate the midpoint of a route for placing the time marker"""
        if not path or len(path) < 2:
            return None
        
        # Get the route geometry
        coords = self.ics.get_path_geometry(path)
        if not coords or len(coords) < 2:
            coords = [(self.ics.nodes[nid][0], self.ics.nodes[nid][1]) for nid in path]
        
        # Find the midpoint coordinate
        mid_idx = len(coords) // 2
        mid_lat, mid_lon = coords[mid_idx]
        
        return (mid_lat, mid_lon)
    
    def create_route_time_marker(self, route):
        """Create a transparent marker showing route time at midpoint"""
        if not self.map_widget or not self.ics:
            return
        
        # Get midpoint of route
        midpoint = self.get_route_midpoint(route['path'])
        if not midpoint:
            return
        
        mid_lat, mid_lon = midpoint
        
        # Format time text
        time_text = f"⏱ {route['cost']:.0f} min"
        
        # Delete existing time marker if any
        if self.route_time_marker:
            self.route_time_marker.delete()
        
        # Create new marker with transparent circle and bold black text
        try:
            self.route_time_marker = self.map_widget.set_marker(
                mid_lat, mid_lon,
                text=time_text,
                marker_color_circle="",           # Transparent marker
                marker_color_outside="",          # Transparent border
                text_color="#000000",             # Black text
                font=("Arial", 11, "bold")        # Bold font
            )
            
            # Try to add white background to text if possible
            if hasattr(self.route_time_marker, 'text_label'):
                self.route_time_marker.text_label.configure(
                    bg='white',
                    relief='solid',
                    borderwidth=2,
                    padx=8,
                    pady=4
                )
        except Exception as e:
            print(f"⚠ Could not create route time marker: {e}")
    
    def initialize_system(self):
        """Initialize or reinitialize the ICS system"""
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            self.ics = ICSSystem(
                model_path=self.current_model_file,
                graph_file=self.current_graph_file,
                osm_file=self.osm_file
            )
            
            # Clear previous state
            self.edge_accidents.clear()  # Changed from node_accidents
            self.all_routes.clear()
            self.route_changes.clear()
            self.route_status.clear()
            self.accident_image_refs.clear()  # Clear image references
            self.accident_frames.clear()  # Clear frame references
            self.accident_edge_paths.clear()  # Clear accident edge paths
            self.selected_accident = None  # Clear selection
            
            # Update file name labels
            self.graph_file_label.config(text=os.path.basename(self.current_graph_file))
            self.model_file_label.config(text=os.path.basename(self.current_model_file))
            
            # Update UI components
            self.update_edge_options()  # Changed from update_node_options
            self.update_accident_listbox()
            self.generate_initial_map()
            
            # Update status
            self.update_path_text(f"System initialized with:\n{os.path.basename(self.current_graph_file)}")
            self.update_results_summary(f"✓ Graph: {os.path.basename(self.current_graph_file)}\n✓ Model: {os.path.basename(self.current_model_file)}\n\nNodes: {len(self.ics.nodes)}\nEdges: {sum(len(v) for v in self.ics.edges.values())}\nOSM Geometry: {len(self.ics.edge_geometry)}\n\nTIP: Try different origin/destination!")
            
            self.root.config(cursor="")
            print("✓ ICS initialized successfully\n")
            
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Initialization Error", f"Failed to initialize system:\n{str(e)}")
            print(f"✗ Initialization failed: {e}")
    
    def update_node_options(self):
        """Update node selection dropdowns based on current graph"""
        self.node_display_to_id = {}
        node_options = []
        
        for nid in sorted(self.ics.node_names.keys()):
            display = f"{nid}: {self.ics.node_names[nid]}"
            node_options.append(display)
            self.node_display_to_id[display] = nid
        
        # Update route selection comboboxes
        self.origin_combo['values'] = node_options
        self.dest_combo['values'] = node_options
        
        # Set defaults
        if len(node_options) > 0:
            self.origin_combo.current(0)
            self.dest_combo.current(min(len(node_options)-1, 10))
    
    def update_edge_options(self):
        """Update edge selection dropdown for accidents"""
        self.edge_display_to_tuple = {}
        edge_options = []
        
        # Create list of all edges
        for from_node, neighbors in self.ics.edges.items():
            for to_node, cost in neighbors:
                # Get node names
                from_name = self.ics.node_names.get(from_node, f"Node {from_node}")
                to_name = self.ics.node_names.get(to_node, f"Node {to_node}")
                
                # Create display string
                display = f"{from_node}→{to_node}: {from_name} → {to_name}"
                edge_options.append(display)
                self.edge_display_to_tuple[display] = (from_node, to_node)
        
        # Sort edges
        edge_options.sort()
        
        # Update accident edge combobox
        self.accident_edge_combo['values'] = edge_options
        
        # Set default
        if len(edge_options) > 0:
            self.accident_edge_combo.current(0)
        
        # Also update node options for route selection
        self.update_node_options()
    
    def select_graph_file(self):
        """Open file dialog to select graph file"""
        filename = filedialog.askopenfilename(
            title="Select Graph File",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialdir="testcases"
        )
        
        if filename:
            self.current_graph_file = filename
            self.graph_file_label.config(text=os.path.basename(filename))
            messagebox.showinfo("Graph File Selected", 
                f"Loading:\n{os.path.basename(filename)}")
            self.initialize_system()
    
    def select_model_file(self):
        """Open file dialog to select ML model"""
        filename = filedialog.askopenfilename(
            title="Select ML Model",
            filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
            initialdir="models"
        )
        
        if filename:
            self.current_model_file = filename
            self.model_file_label.config(text=os.path.basename(filename))
            messagebox.showinfo("Model Selected", 
                f"Loading:\n{os.path.basename(filename)}")
            self.initialize_system()
    
    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=0)  # Control panel: fixed width
        main_frame.columnconfigure(1, weight=1)  # Map: takes remaining space
        main_frame.rowconfigure(0, weight=1)
        
        # === LEFT PANEL ===
        control_frame = ttk.LabelFrame(main_frame, text="⚙️ Control Panel", padding="1")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.S), padx=5)
        
        # 0. File Selection
        file_frame = ttk.LabelFrame(control_frame, text="📁 Configuration", padding="5")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.N, tk.E), pady=5)
        
        ttk.Button(file_frame, text=" 📄 Select Graph File (.txt) ", 
                  command=self.select_graph_file).grid(row=0, column=0, pady=3, sticky=tk.W)
        
        self.graph_file_label = ttk.Label(file_frame, text="heritage_assignment_15.txt", 
                                         font=('Arial', 8), foreground='#666666')
        self.graph_file_label.grid(row=0, column=1, padx=(10, 0), sticky=tk.W)
        
        ttk.Button(file_frame, text=" 🧠 Select ML Model (.h5) ", 
                  command=self.select_model_file).grid(row=1, column=0, pady=3, sticky=tk.W)
        
        self.model_file_label = ttk.Label(file_frame, text="model2_mobilenetV2.h5",
                                         font=('Arial', 8), foreground='#666666')
        self.model_file_label.grid(row=1, column=1, padx=(10, 0), sticky=tk.W)
        
        # 1. Route Selection
        route_frame = ttk.LabelFrame(control_frame, text="🗺️ Route Selection", padding="5")
        route_frame.grid(row=1, column=0, sticky=(tk.W, tk.N, tk.E), pady=5)
        
        ttk.Label(route_frame, text="Origin Node:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.origin_combo = ttk.Combobox(route_frame, width=35, state='readonly')
        self.origin_combo.bind('<<ComboboxSelected>>', self.on_route_change)
        self.origin_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Label(route_frame, text="Destination:", font=('Arial', 9, 'bold')).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.dest_combo = ttk.Combobox(route_frame, width=35, state='readonly')
        self.dest_combo.bind('<<ComboboxSelected>>', self.on_route_change)
        self.dest_combo.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        # 2. Accident Detection - FIXED: Removed S sticky to prevent vertical expansion
        accident_frame = ttk.LabelFrame(control_frame, text="🚨 Accident Detection", padding="5")
        accident_frame.grid(row=2, column=0, sticky=(tk.W, tk.N), pady=5)
        
        ttk.Label(accident_frame, text="Select Edge:", font=('Arial', 9, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.accident_edge_combo = ttk.Combobox(accident_frame, width=45, state='readonly')
        self.accident_edge_combo.grid(row=0, column=1, sticky=tk.W, pady=5, padx=5)
        
        ttk.Button(accident_frame, text="📷 Add Accident Image", 
                  command=self.add_accident).grid(row=1, column=0, columnspan=2, pady=5, sticky=tk.W)
        
        # MODIFIED: Frame with canvas for scrollable accident list with images
        list_container = ttk.Frame(accident_frame, width=50)
        list_container.grid(row=2, column=0, columnspan=2, pady=5)
        list_container.grid_propagate(False)  # Prevent frame from resizing to contents
        
        # Canvas for scrollable content - FIXED: Reduced height
        self.accident_canvas = tk.Canvas(list_container, height=140, bg='white', 
                                         highlightthickness=1, highlightbackground='#cccccc')
        scrollbar = ttk.Scrollbar(list_container, orient="vertical", command=self.accident_canvas.yview)
        
        # Frame inside canvas to hold accident entries
        self.accident_list_frame = ttk.Frame(self.accident_canvas)
        
        self.accident_list_frame.bind(
            "<Configure>",
            lambda e: self.accident_canvas.configure(scrollregion=self.accident_canvas.bbox("all"))
        )
        
        self.accident_canvas.create_window((0, 0), window=self.accident_list_frame, anchor="nw")
        self.accident_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.accident_canvas.pack(side=tk.LEFT, fill=tk.Y)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        btn_frame = ttk.Frame(accident_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=5, sticky=tk.W)
        ttk.Button(btn_frame, text="❌ Remove", command=self.remove_accident).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        ttk.Button(btn_frame, text="🗑️ Clear All", command=self.clear_all_accidents).pack(side=tk.LEFT, padx=2, expand=True, fill=tk.X)
        
        # 3. Actions
        action_frame = ttk.LabelFrame(control_frame, text="🔍 Actions", padding="5")
        action_frame.grid(row=3, column=0, sticky=(tk.W, tk.N, tk.E), pady=5)
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        action_frame.columnconfigure(2, weight=1)
        
        self.btn_all_routes = ttk.Button(action_frame, text="All Routes", command=self.find_all_routes)
        self.btn_all_routes.grid(row=0, column=0, padx=2, sticky=(tk.W, tk.E))
        
        self.btn_best_routes = ttk.Button(action_frame, text="Best Routes", command=self.show_best_3, state='disabled')
        self.btn_best_routes.grid(row=0, column=1, padx=2, sticky=(tk.W, tk.E))
        
        self.btn_reset = ttk.Button(action_frame, text="Reset Map", command=self.reset_map_action)
        self.btn_reset.grid(row=0, column=2, padx=2, sticky=(tk.W, tk.E))
        
        # Path Details Window
        ttk.Label(action_frame, text="Path Details:", font=('Arial', 8, 'bold')).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(10, 2))
        path_scroll = ttk.Scrollbar(action_frame)
        path_scroll.grid(row=2, column=2, sticky='ns')
        self.path_text = tk.Text(action_frame, height=6, width=50, yscrollcommand=path_scroll.set, 
                                font=('Arial', 8), wrap=tk.WORD, bg="#f4f4f4", relief="flat")
        self.path_text.grid(row=2, column=0, columnspan=2, sticky='we')
        path_scroll.config(command=self.path_text.yview)
        action_frame.columnconfigure(0, weight=10)
        
        # NEW: Best Routes Section (initially hidden) - MOVED to separate row
        self.best_routes_frame = ttk.LabelFrame(control_frame, text="🏆 Best Routes", padding="5")
        self.best_routes_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N), pady=2)
        self.best_routes_frame.grid_remove()  # Hide initially
        
        # Container for best route checkboxes
        self.best_routes_container = ttk.Frame(self.best_routes_frame)
        self.best_routes_container.pack(fill=tk.BOTH, expand=True)
        
        # 4. Results Summary with Per-Route Checkboxes
        results_frame = ttk.LabelFrame(control_frame, text="📊 Results Summary (Toggle Routes)", padding="1")
        results_frame.grid(row=5, column=0, sticky=(tk.W, tk.N, tk.S, tk.E), pady=1)
        control_frame.rowconfigure(5, weight=1)
        
        # Create scrollable frame for route checkboxes
        self.results_canvas = tk.Canvas(results_frame, highlightthickness=0, width=400)
        result_scroll = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_canvas.yview)
        self.results_scrollable_frame = ttk.Frame(self.results_canvas)
        
        self.results_scrollable_frame.bind(
            "<Configure>",
            lambda e: self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))
        )
        
        self.results_canvas.create_window((0, 0), window=self.results_scrollable_frame, anchor="nw")
        self.results_canvas.configure(yscrollcommand=result_scroll.set)
        
        self.results_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.BOTH)
        
        # === RIGHT PANEL ===
        map_frame = ttk.LabelFrame(main_frame, text="🗺️ Interactive Route Visualization", padding="10")
        map_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        map_frame.rowconfigure(2, weight=1)
        map_frame.columnconfigure(0, weight=1)
        
        self.map_status_label = ttk.Label(map_frame, text="Loading map...", font=('Arial', 10))
        self.map_status_label.grid(row=0, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Algorithm-level visibility controls
        algo_control_frame = ttk.LabelFrame(map_frame, text="🎨 Route Visibility (Algorithm Level)", padding="5")
        algo_control_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E))
        
        self.algo_visibility = {}
        
        for i, algo in enumerate(['A*', 'BFS', 'DFS', 'GBFS', 'HPA*', 'FNS']):
            # Default: Only A* is checked
            default_checked = (algo == 'A*')
            var = tk.BooleanVar(value=default_checked)
            self.algo_visibility[algo] = var
            
            cb_frame = ttk.Frame(algo_control_frame)
            cb_frame.grid(row=0, column=i, padx=5, pady=3, sticky=tk.W)
            
            # Color indicator
            base_color = self.algo_base_colors[algo]
            color_hex = f'#{base_color[0]:02x}{base_color[1]:02x}{base_color[2]:02x}'
            canvas = tk.Canvas(cb_frame, width=20, height=10, highlightthickness=0, bg='white')
            canvas.grid(row=0, column=0, padx=(0, 5))
            canvas.create_rectangle(0, 0, 20, 10, fill=color_hex, outline='black', width=1)
            
            cb_widget = ttk.Checkbutton(
                cb_frame, 
                text=algo, 
                variable=var, 
                command=lambda a=algo: self.on_algo_visibility_change(a)
            )
            cb_widget.grid(row=0, column=1, sticky=tk.W)
            
            # Store checkbox widget for enabling/disabling later
            if not hasattr(self, 'algo_checkbuttons'):
                self.algo_checkbuttons = {}
            self.algo_checkbuttons[algo] = cb_widget
        
        # Map Widget
        if HAS_MAP:
            try:
                self.map_widget = tkintermapview.TkinterMapView(map_frame, corner_radius=0)
                self.map_widget.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
                self.map_widget.set_zoom(17)
                try: 
                    self.map_widget.set_tile_server("https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png", max_zoom=19)
                except: 
                    pass
            except Exception as e:
                print(f"✗ Error creating map: {e}")
                self.map_widget = None
                self.create_fallback(map_frame)
        else:
            self.map_widget = None
            self.create_fallback(map_frame)
    
    def create_fallback(self, parent):
        fallback = ttk.Frame(parent, relief='solid', borderwidth=1)
        fallback.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        tk.Label(fallback, text="Map Not Available\nInstall: pip install tkintermapview").pack(pady=50)
    
    def update_path_text(self, text_content):
        self.path_text.config(state=tk.NORMAL)
        self.path_text.delete('1.0', tk.END)
        self.path_text.insert(tk.END, text_content)
        self.path_text.config(state=tk.DISABLED)
    
    def update_results_summary(self, text_content):
        """Update results summary with plain text"""
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()
        
        label = tk.Label(
            self.results_scrollable_frame, 
            text=text_content, 
            font=('Courier', 9),
            justify=tk.LEFT,
            anchor='w',
            bg='white'
        )
        label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def populate_route_checkboxes(self):
        """Populate Results Summary with color-coded route checkboxes"""
        for widget in self.results_scrollable_frame.winfo_children():
            widget.destroy()
        
        self.route_checkboxes.clear()
        self.route_visibility.clear()
        
        if not self.all_routes:
            label = tk.Label(
                self.results_scrollable_frame, 
                text="No routes found yet.\nClick 'All Routes' to begin.",
                font=('Arial', 9),
                fg='gray',
                bg='white'
            )
            label.pack(pady=20)
            return
        
        routes_by_algo = {}
        for route in self.all_routes:
            algo = route['algo']
            if algo not in routes_by_algo:
                routes_by_algo[algo] = []
            routes_by_algo[algo].append(route)
        
        for algo in ['A*', 'BFS', 'DFS', 'GBFS', 'HPA*', 'FNS']:
            if algo not in routes_by_algo:
                continue
            
            routes = routes_by_algo[algo]
            total_routes = len(routes)
            
            header_frame = ttk.Frame(self.results_scrollable_frame)
            header_frame.pack(fill=tk.X, padx=5, pady=(10, 2))
            
            ttk.Label(
                header_frame, 
                text=f"--- {algo} ({total_routes} routes) ---",
                font=('Arial', 9, 'bold')
            ).pack(anchor='w')
            
            for route in routes:
                route_id = route['id']
                route_key = (algo, route_id)
                
                color = self.get_route_color(algo, route_id, total_routes)
                
                cb_frame = ttk.Frame(self.results_scrollable_frame)
                cb_frame.pack(fill=tk.X, padx=10, pady=1)
                
                color_canvas = tk.Canvas(cb_frame, width=30, height=12, highlightthickness=0, bg='white')
                color_canvas.pack(side=tk.LEFT, padx=(0, 5))
                color_canvas.create_rectangle(0, 0, 30, 12, fill=color, outline='black', width=1)
                
                # Default: Only Route #1 for each algorithm is checked
                default_checked = (route_id == 1)
                var = tk.BooleanVar(value=default_checked)
                self.route_visibility[route_key] = var
                
                path_str = " → ".join(str(n) for n in route['path'][:6])
                if len(route['path']) > 6:
                    path_str += "..."
                
                status_icon = self.route_status.get(route_key, "➡️")
                label_text = f"Route #{route_id} {status_icon} | {route['cost']:.1f}min | {path_str}"
                
                cb = ttk.Checkbutton(
                    cb_frame,
                    text=label_text,
                    variable=var,
                    command=lambda a=algo, rid=route_id: self.on_route_checkbox_click(a, rid)
                )
                cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
                
                # Add expand button to show full path
                expand_btn = tk.Button(
                    cb_frame, 
                    text="➕", 
                    font=('Arial', 8),
                    width=2,
                    relief=tk.FLAT,
                    bg='#e0e0e0',
                    command=lambda r=route: self.show_route_details(r)
                )
                expand_btn.pack(side=tk.LEFT, padx=(2, 0))
                
                self.route_checkboxes[route_key] = (cb, var)
        
        legend_frame = ttk.Frame(self.results_scrollable_frame)
        legend_frame.pack(fill=tk.X, padx=5, pady=(10, 5))
        
        if self.edge_accidents:
            legend_text = "🔄 = Route changed after accident\n➡️ = Route unchanged"
        else:
            legend_text = "➡️ = No accidents detected"
        
        ttk.Label(
            legend_frame,
            text=legend_text,
            font=('Arial', 8),
            foreground='gray'
        ).pack(anchor='w')
    
    def show_route_details(self, route):
        """Display full route path in a popup window"""
        popup = tk.Toplevel(self.root)
        popup.title(f"{route['algo']} Route #{route['id']} Details")
        popup.geometry("500x330")
        
        # Main frame with padding
        main_frame = ttk.Frame(popup, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header info
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(
            header_frame,
            text=f"Algorithm: {route['algo']} | Route #{route['id']}",
            font=('Arial', 11, 'bold')
        ).pack(anchor='w')
        
        ttk.Label(
            header_frame,
            text=f"Total Time: {route['cost']:.2f}min",
            font=('Arial', 10)
        ).pack(anchor='w')
        
        ttk.Label(
            header_frame,
            text=f"Number of Nodes: {len(route['path'])}",
            font=('Arial', 10)
        ).pack(anchor='w')
        
        # Full path display with scrollbar
        path_frame = ttk.LabelFrame(main_frame, text="Complete Path", padding="5")
        path_frame.pack(fill=tk.BOTH, expand=True)
        
        path_text = tk.Text(path_frame, wrap=tk.WORD, font=('Courier', 9), height=10)
        scrollbar = ttk.Scrollbar(path_frame, orient=tk.VERTICAL, command=path_text.yview)
        path_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        path_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Format the full path
        full_path = " → ".join(str(n) for n in route['path'])
        path_text.insert('1.0', full_path)
        path_text.config(state=tk.DISABLED)  # Make read-only
        
        # Close button
        close_btn = ttk.Button(main_frame, text="Close", command=popup.destroy)
        close_btn.pack(pady=(10, 0))
    
    def on_algo_visibility_change(self, algo):
        """When algorithm checkbox is toggled, enable/disable its route checkboxes"""
        is_visible = self.algo_visibility[algo].get()
        
        for route_key, (cb, var) in self.route_checkboxes.items():
            if route_key[0] == algo:
                if is_visible:
                    cb.config(state='normal')
                else:
                    cb.config(state='disabled')
                    var.set(False)
        
        self.update_route_visibility()
    
    def on_route_checkbox_click(self, algo, route_id):
        """Handle route checkbox click - ensure only one route per algorithm is checked"""
        clicked_route_key = (algo, route_id)
        
        # If this route is being checked, uncheck all other routes for this algorithm
        if self.route_visibility[clicked_route_key].get():
            for route_key, var in self.route_visibility.items():
                if route_key[0] == algo and route_key != clicked_route_key:
                    var.set(False)
        
        self.update_route_visibility()
    
    def reset_map_action(self):
        self.generate_initial_map()
        self.update_path_text("Map reset.")
        self.update_results_summary(f"Map cleared.\n\nGraph: {os.path.basename(self.current_graph_file)}\nModel: {os.path.basename(self.current_model_file)}")
        self.btn_best_routes.config(state='disabled')
        self.all_routes = []
        self.route_changes = {}
        self.route_status.clear()
    
    def on_route_change(self, event=None):
        if not self.ics:
            return
        self.generate_initial_map()
        self.update_path_text("Route points changed.")
        self.all_routes = []
        self.route_changes = {}
        self.route_status.clear()
        self.update_results_summary("Origin/Destination changed.\nClick 'All Routes' to find paths.")
    
    def generate_initial_map(self):
        if not self.map_widget or not self.ics: 
            return
        try:
            self.map_widget.delete_all_marker()
            self.map_widget.delete_all_path()
            self.node_markers.clear()
            self.edge_paths.clear()
            self.route_paths.clear()
            self.accident_edge_paths.clear()
            
            # Clear route time marker
            if self.route_time_marker:
                self.route_time_marker.delete()
                self.route_time_marker = None
            
            origin = self.node_display_to_id[self.origin_combo.get()]
            destination = self.node_display_to_id[self.dest_combo.get()]
            
            lats = [coord[0] for coord in self.ics.nodes.values()]
            lons = [coord[1] for coord in self.ics.nodes.values()]
            self.map_widget.set_position(sum(lats)/len(lats) - 0.0008, sum(lons)/len(lons))
            
            # Draw all edges (gray by default)
            for from_node, neighbors in self.ics.edges.items():
                for to_node, cost in neighbors:
                    edge_coords = self.ics.get_path_geometry([from_node, to_node])
                    if not edge_coords or len(edge_coords) < 2:
                        edge_coords = [(self.ics.nodes[from_node]), (self.ics.nodes[to_node])]
                    path = self.map_widget.set_path(edge_coords, color="#999999", width=2)
                    self.edge_paths.append(path)
            
            # Highlight accident edges in RED (only if severity is not "none")
            for edge_tuple, info in self.edge_accidents.items():
                severity = info['severity']
                # Only highlight if accident is minor, moderate, or severe (not none)
                if severity > 0:  # 0 = none, 1+ = minor/moderate/severe
                    from_node, to_node = edge_tuple
                    edge_coords = self.ics.get_path_geometry([from_node, to_node])
                    if not edge_coords or len(edge_coords) < 2:
                        edge_coords = [(self.ics.nodes[from_node]), (self.ics.nodes[to_node])]
                    
                    # Draw accident edge in RED with thicker width
                    accident_path = self.map_widget.set_path(edge_coords, color="#FF0000", width=5)
                    self.accident_edge_paths[edge_tuple] = accident_path
            
            # Draw node markers
            for node_id, (lat, lon) in self.ics.nodes.items():
                marker_color = "#B9B9B9"
                node_name = self.ics.node_names.get(node_id, "")
                text_label = f"{node_id}: {node_name}"
                if node_id == origin:
                    marker_color = "#00cc00"; text_label = f"START: {node_id}\n{node_name}"
                elif node_id == destination:
                    marker_color = "#f20101"; text_label = f"END: {node_id}\n{node_name}"
                
                marker = self.map_widget.set_marker(lat, lon, text=text_label, marker_color_circle=marker_color, 
                                                   marker_color_outside="#929292", text_color="#000000", font=("Arial", 8))
                self.node_markers[node_id] = marker
            
            self.map_status_label.config(text=f"📍 Origin: {origin} | Dest: {destination}")
        except Exception as e:
            print(f"✗ Error generating map: {e}")

    def update_route_visibility(self):
        """Update map based on individual route visibility checkboxes"""
        if not self.map_widget or not self.ics: 
            return
        
        try:
            for path_list in self.route_paths.values():
                if isinstance(path_list, list):
                    for path in path_list: 
                        path.delete()
                else:
                    path_list.delete()
            self.route_paths.clear()
            
            # Clear existing time marker
            if self.route_time_marker:
                self.route_time_marker.delete()
                self.route_time_marker = None
            
            visible_count = 0
            selected_route = None  # Track which route to show time for
            
            for route in self.all_routes:
                algo = route['algo']
                route_id = route['id']
                route_key = (algo, route_id)
                
                if route_key in self.route_visibility and self.route_visibility[route_key].get():
                    if self.algo_visibility[algo].get():
                        routes_for_algo = [r for r in self.all_routes if r['algo'] == algo]
                        color = self.get_route_color(algo, route_id, len(routes_for_algo))
                        
                        coords = self.ics.get_path_geometry(route['path'])
                        if not coords or len(coords) < 2:
                            coords = [(self.ics.nodes[nid][0], self.ics.nodes[nid][1]) for nid in route['path']]
                        
                        width = 5 if route_id == 1 else 3
                        path = self.map_widget.set_path(coords, color=color, width=width)
                        
                        if route_key not in self.route_paths:
                            self.route_paths[route_key] = []
                        self.route_paths[route_key].append(path)
                        
                        visible_count += 1
                        
                        # Remember the first visible route for time marker
                        if selected_route is None:
                            selected_route = route
            
            # IMPORTANT: Redraw accident edges on top so they're never covered
            self.redraw_accident_edges_on_top()
            
            # Add time marker for the selected route
            if selected_route:
                self.create_route_time_marker(selected_route)
            
            self.map_status_label.config(text=f"📍 {visible_count} routes visible")
            
        except Exception as e:
            print(f"✗ Error updating visibility: {e}")
    
    def redraw_accident_edges_on_top(self):
        """Redraw accident edges on top layer so they're always visible"""
        if not self.map_widget or not self.ics:
            return
        
        # Delete existing accident edge paths
        for path in self.accident_edge_paths.values():
            if path:
                path.delete()
        self.accident_edge_paths.clear()
        
        # Redraw accident edges
        for edge_tuple, info in self.edge_accidents.items():
            severity = info['severity']
            if severity > 0:  # Only minor/moderate/severe
                from_node, to_node = edge_tuple
                edge_coords = self.ics.get_path_geometry([from_node, to_node])
                if not edge_coords or len(edge_coords) < 2:
                    edge_coords = [(self.ics.nodes[from_node]), (self.ics.nodes[to_node])]
                
                # Draw RED accident edge on top
                accident_path = self.map_widget.set_path(edge_coords, color="#FF0000", width=6)
                self.accident_edge_paths[edge_tuple] = accident_path

    def add_accident(self):
        if not self.ics:
            messagebox.showwarning("System Not Ready", "Please load a graph file first.")
            return
        
        # Get selected edge
        edge_tuple = self.edge_display_to_tuple[self.accident_edge_combo.get()]
        from_node, to_node = edge_tuple
        
        image_path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png"), ("All", "*.*")])
        if not image_path: 
            return
        try:
            self.root.config(cursor="wait")
            self.root.update()
            
            # Detect accident using ML model
            prediction = self.ics.detect_accident_at_node(from_node, image_path)  # Still use ICS method
            
            # Store accident for edge
            self.edge_accidents[edge_tuple] = {
                'severity': prediction['severity'], 
                'image_path': image_path, 
                'prediction': prediction
            }
            
            self.update_accident_listbox()
            self.generate_initial_map()
            self.root.config(cursor="")
            
            # Get node names for display
            from_name = self.ics.node_names.get(from_node, f"Node {from_node}")
            to_name = self.ics.node_names.get(to_node, f"Node {to_node}")
            
            messagebox.showinfo(
                "Accident Detected", 
                f"Edge {from_node}→{to_node}: {from_name} → {to_name}\n{prediction['class_name']}\nConfidence: {prediction['confidence']:.1%}"
            )
            
            if self.all_routes: 
                self.find_all_routes()
        except Exception as e:
            self.root.config(cursor="")
            messagebox.showerror("Error", f"Failed:\n{str(e)}")

    def update_accident_listbox(self):
        """NEW: Update accident list with image thumbnails and click-to-select (now for edges)"""
        # Clear existing entries
        for widget in self.accident_list_frame.winfo_children():
            widget.destroy()
        
        # Clear old image references
        self.accident_image_refs.clear()
        self.accident_frames.clear()
        
        # Reset selection if the selected accident was removed
        if self.selected_accident and self.selected_accident not in self.edge_accidents:
            self.selected_accident = None
        
        # Create entries for each accident (now edge-based)
        for idx, (edge_tuple, info) in enumerate(sorted(self.edge_accidents.items())):
            from_node, to_node = edge_tuple
            
            # Create clickable frame
            entry_frame = tk.Frame(self.accident_list_frame, relief='solid', borderwidth=1, 
                                  bg='white', cursor='hand2')
            entry_frame.pack(fill=tk.X, padx=2, pady=2)
            
            # Store frame reference
            self.accident_frames[edge_tuple] = entry_frame
            
            # Load and resize image thumbnail
            try:
                img = Image.open(info['image_path'])
                img.thumbnail((80, 80), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Store reference to prevent garbage collection
                self.accident_image_refs[edge_tuple] = photo
                
                # Image label
                img_label = tk.Label(entry_frame, image=photo, bg='white')
                img_label.pack(side=tk.LEFT, padx=(5, 5))
            except Exception as e:
                print(f"⚠ Could not load thumbnail for edge {from_node}→{to_node}: {e}")
                # Placeholder if image fails to load
                img_label = tk.Label(entry_frame, text="🖼️", font=('Arial', 20), bg='white')
                img_label.pack(side=tk.LEFT, padx=(5, 5))
            
            # Text label with edge and severity
            text_label = tk.Label(
                entry_frame,
                text=f"Edge {from_node}→{to_node}: {info['prediction']['class_name']}",
                font=('Courier', 8),
                anchor='w',
                bg='white'
            )
            text_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 5))
            
            # Bind click event to frame and all children
            entry_frame.bind('<Button-1>', lambda e, edge=edge_tuple: self.select_accident(edge))
            img_label.bind('<Button-1>', lambda e, edge=edge_tuple: self.select_accident(edge))
            text_label.bind('<Button-1>', lambda e, edge=edge_tuple: self.select_accident(edge))
            
            # Highlight if this is the selected accident
            if edge_tuple == self.selected_accident:
                entry_frame.config(bg='#cce5ff', relief='solid', borderwidth=2)
                img_label.config(bg='#cce5ff')
                text_label.config(bg='#cce5ff')
    
    def select_accident(self, edge_tuple):
        """Handle accident selection (now edge-based)"""
        # Update selection
        self.selected_accident = edge_tuple
        
        # Update visual highlighting
        for edge, frame in self.accident_frames.items():
            if edge == edge_tuple:
                # Highlight selected
                frame.config(bg='#cce5ff', relief='solid', borderwidth=2)
                for child in frame.winfo_children():
                    child.config(bg='#cce5ff')
            else:
                # Unhighlight others
                frame.config(bg='white', relief='solid', borderwidth=1)
                for child in frame.winfo_children():
                    child.config(bg='white')

    def remove_accident(self):
        """Remove the selected accident (now edge-based)"""
        if not self.ics:
            return
        
        if not self.edge_accidents:
            messagebox.showinfo("No Accidents", "No accidents to remove.")
            return
        
        if self.selected_accident is None:
            messagebox.showinfo("No Selection", "Please click on an accident to select it first.")
            return
        
        # Remove the selected accident
        edge_tuple = self.selected_accident
        if edge_tuple in self.edge_accidents:
            del self.edge_accidents[edge_tuple]
        
        # Clean up image reference
        if edge_tuple in self.accident_image_refs:
            del self.accident_image_refs[edge_tuple]
        
        # Clear selection
        self.selected_accident = None
        
        self.update_accident_listbox()
        self.generate_initial_map()
        
        if self.all_routes: 
            self.find_all_routes()

    def clear_all_accidents(self):
        if not self.ics:
            return
        self.edge_accidents.clear()  # Changed from node_accidents
        self.accident_image_refs.clear()
        self.accident_frames.clear()
        self.accident_edge_paths.clear()  # Clear accident edge paths
        self.selected_accident = None  # Clear selection
        self.update_accident_listbox()
        self.generate_initial_map()
        if self.all_routes: 
            self.find_all_accidents()

    def find_all_routes(self):
        if not self.ics:
            messagebox.showwarning("System Not Ready", "Please load a graph file first.")
            return
        
        # Switch to all routes mode if in best routes mode
        if self.viewing_mode == 'best_routes':
            self.switch_to_all_routes_mode()
            
        origin = self.node_display_to_id[self.origin_combo.get()]
        destination = self.node_display_to_id[self.dest_combo.get()]
        if origin == destination:
            messagebox.showwarning("Invalid", "Origin/Destination same.")
            return
        
        self.root.config(cursor="wait")
        self.root.update()
        
        old_routes_by_key = {}
        if self.all_routes:
            for route in self.all_routes:
                route_key = (route['algo'], route['id'])
                old_routes_by_key[route_key] = route['path']
        
        # Convert edge_accidents to node_accidents format for ICS system
        # ICS expects node-based accidents, but we store edge-based
        # Convert by using the "from_node" of each edge
        node_accidents_for_ics = {}
        for (from_node, to_node), info in self.edge_accidents.items():
            # Use from_node as the accident location for ICS
            if from_node not in node_accidents_for_ics:
                node_accidents_for_ics[from_node] = info
        
        self.ics.update_travel_times_for_node(node_accidents_for_ics)
        
        algorithms_to_run = {
            'A*': astar_search,
            'BFS': bfs_search,
            'DFS': dfs_search,
            'GBFS': gbfs_search,
            'HPA*': hpa_star_search,
            'FNS': fns_search
        }
        
        self.all_routes = []
        self.route_status.clear()
        
        for algo_name, algo_func in algorithms_to_run.items():
            routes = self.ics.find_k_routes(algo_name, algo_func, origin, destination, k=5)
            self.all_routes.extend(routes)
            
            for route in routes:
                route_key = (route['algo'], route['id'])
                
                if route_key in old_routes_by_key:
                    if route['path'] == old_routes_by_key[route_key]:
                        self.route_status[route_key] = "➡️"
                    else:
                        self.route_status[route_key] = "🔄"
                else:
                    self.route_status[route_key] = "➡️"
        
        self.root.config(cursor="")
        
        if not self.all_routes:
            self.update_results_summary("❌ No routes found!")
            return

        self.populate_route_checkboxes()
        self.visualize_all_routes()
        self.btn_best_routes.config(state='normal')
        
        summary = f"✓ Found {len(self.all_routes)} unique routes\n\n"
        for algo in ['A*', 'BFS', 'DFS', 'GBFS', 'HPA*', 'FNS']:
            algo_routes = [r for r in self.all_routes if r['algo'] == algo]
            if algo_routes:
                summary += f"{algo}: {len(algo_routes)} routes\n"
        self.update_path_text(summary)

    def visualize_all_routes(self):
        self.update_route_visibility()

    def show_best_3(self):
        """Show top 3 best routes with different paths"""
        if len(self.all_routes) == 0: 
            return
        
        # Switch to best routes mode
        self.viewing_mode = 'best_routes'
        
        # Find top 3 routes with DIFFERENT paths
        sorted_routes = sorted(self.all_routes, key=lambda x: x['cost'])
        self.best_routes = []
        seen_paths = set()
        
        for route in sorted_routes:
            path_tuple = tuple(route['path'])
            if path_tuple not in seen_paths:
                self.best_routes.append(route)
                seen_paths.add(path_tuple)
                if len(self.best_routes) == 3:
                    break
        
        # Disable algorithm and results summary checkboxes
        for algo, cb_widget in self.algo_checkbuttons.items():
            cb_widget.config(state='disabled')
        
        for route_key, (cb, var) in self.route_checkboxes.items():
            cb.config(state='disabled')
        
        # Show Best Routes UI
        self.best_routes_frame.grid()
        
        # Populate best routes checkboxes
        self.populate_best_routes_checkboxes()
        
        # Update Path Details
        text_content = "🏆 BEST ROUTES MODE\n\n"
        text_content += "Showing top 3 routes with different paths.\n"
        text_content += "Check a route below to highlight it on the map.\n"
        self.update_path_text(text_content)
        
        # Update map to show best routes
        self.update_best_routes_visibility()
    
    def populate_best_routes_checkboxes(self):
        """Create checkboxes for best 3 routes"""
        # Clear existing
        for widget in self.best_routes_container.winfo_children():
            widget.destroy()
        
        self.best_route_checkboxes.clear()
        self.best_route_visibility.clear()
        
        if not self.best_routes:
            return
        
        for i, route in enumerate(self.best_routes, 1):
            frame = ttk.Frame(self.best_routes_container)
            frame.pack(fill=tk.X, padx=5, pady=2)
            
            # Default: only 1st is checked
            var = tk.BooleanVar(value=(i == 1))
            self.best_route_visibility[i] = var
            
            # Create label text
            path_str = " → ".join(str(n) for n in route['path'][:6])
            if len(route['path']) > 6:
                path_str += "..."
            
            label_text = f"{i}. [{route['algo']} #{route['id']}] {route['cost']:.1f}min | {path_str}"
            
            cb = ttk.Checkbutton(
                frame,
                text=label_text,
                variable=var,
                command=lambda rank=i: self.on_best_route_checkbox_click(rank)
            )
            cb.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            self.best_route_checkboxes[i] = (cb, var)
    
    def on_best_route_checkbox_click(self, selected_rank):
        """Handle best route checkbox - only one can be checked at a time"""
        # Uncheck all others
        for rank, var in self.best_route_visibility.items():
            if rank != selected_rank:
                var.set(False)
        
        # If user unchecked the selected one, recheck it (at least one must be checked)
        if not self.best_route_visibility[selected_rank].get():
            self.best_route_visibility[selected_rank].set(True)
        
        self.update_best_routes_visibility()
    
    def update_best_routes_visibility(self):
        """Update map for best routes mode"""
        if not self.map_widget or not self.ics:
            return
        
        try:
            # Clear all route paths
            for path_list in self.route_paths.values():
                if isinstance(path_list, list):
                    for path in path_list:
                        path.delete()
                else:
                    path_list.delete()
            self.route_paths.clear()
            
            # Clear gray paths
            for path in self.best_routes_gray_paths:
                path.delete()
            self.best_routes_gray_paths.clear()
            
            # Clear existing time marker
            if self.route_time_marker:
                self.route_time_marker.delete()
                self.route_time_marker = None
            
            # Find which route is selected
            selected_rank = None
            for rank, var in self.best_route_visibility.items():
                if var.get():
                    selected_rank = rank
                    break
            
            # STEP 1: Draw gray routes FIRST (bottom layer)
            for rank, route in enumerate(self.best_routes, 1):
                if rank != selected_rank:  # Only draw unselected routes
                    coords = self.ics.get_path_geometry(route['path'])
                    if not coords or len(coords) < 2:
                        coords = [(self.ics.nodes[nid][0], self.ics.nodes[nid][1]) for nid in route['path']]
                    
                    # Draw in light gray
                    path = self.map_widget.set_path(coords, color="#CCCCCC", width=3)
                    self.best_routes_gray_paths.append(path)
            
            # STEP 2: Draw selected route LAST (top layer)
            selected_route = None
            if selected_rank:
                selected_route = self.best_routes[selected_rank - 1]
                coords = self.ics.get_path_geometry(selected_route['path'])
                if not coords or len(coords) < 2:
                    coords = [(self.ics.nodes[nid][0], self.ics.nodes[nid][1]) for nid in selected_route['path']]
                
                # Draw in full color on top
                algo = selected_route['algo']
                routes_for_algo = [r for r in self.all_routes if r['algo'] == algo]
                color = self.get_route_color(algo, selected_route['id'], len(routes_for_algo))
                path = self.map_widget.set_path(coords, color=color, width=6)
                self.route_paths[('best', selected_rank)] = [path]
            
            # STEP 3: Redraw accident edges on very top
            self.redraw_accident_edges_on_top()
            
            # STEP 4: Add time marker for selected route
            if selected_route:
                self.create_route_time_marker(selected_route)
            
            self.map_status_label.config(text=f"🏆 Best Routes Mode - Showing route #{selected_rank}")
            
        except Exception as e:
            print(f"✗ Error updating best routes visibility: {e}")
    
    def switch_to_all_routes_mode(self):
        """Switch from best routes mode back to all routes mode"""
        self.viewing_mode = 'all_routes'
        
        # Hide Best Routes UI
        self.best_routes_frame.grid_remove()
        
        # Clear best routes data
        self.best_routes.clear()
        self.best_route_checkboxes.clear()
        self.best_route_visibility.clear()
        for path in self.best_routes_gray_paths:
            if path:
                path.delete()
        self.best_routes_gray_paths.clear()
        
        # Re-enable algorithm and results summary checkboxes
        for algo, cb_widget in self.algo_checkbuttons.items():
            cb_widget.config(state='normal')
        
        for route_key, (cb, var) in self.route_checkboxes.items():
            cb.config(state='normal')
        
        # Update map
        self.update_route_visibility()


# === MAIN ===
if __name__ == '__main__':
    print("="*70)
    print("ICS GUI - WITH IMAGE PREVIEW")
    print("  Features: Individual route toggle + Image thumbnails")
    print("  Group: SkyNet | Assignment 2B")
    print("="*70)
    root = tk.Tk()
    app = EnhancedICS_GUI(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\nClosed.")
    except Exception as e:
        print(f"\n Error: {e}")
        import traceback
        traceback.print_exc()
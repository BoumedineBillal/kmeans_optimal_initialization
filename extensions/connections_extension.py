import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from ..utils.distance_utils import (
    calculate_point_distances, 
    generate_connections,
    filter_connections_by_count,
    filter_connections_by_direction
)
from ..utils.visualization import create_connections_window

class ConnectionsExtension:
    """Extension to add connections functionality to the K-means app"""
    
    def __init__(self, parent_app):
        """
        Initialize the extension
        
        Args:
            parent_app: The main K-means application (KMeansClusteringApp instance)
        """
        self.app = parent_app
        
        # Default parameters
        self.distance_std_factor = 0.71  # Multiplier for distance std threshold
        self.min_connections_factor = 1.0  # Factor for minimum connections
        self.direction_std_factor = 2.0  # Factor for direction filtering
        self.nfd = 2  # Number of times to apply direction filtering
        
        # For tracking last used connections
        self.last_connections = None
        self.last_points = None
        
        # Add UI elements to the main app
        self.add_ui_elements()
    
    def add_ui_elements(self):
        """Add UI elements to the parent application"""
        # Add content directly to the right frame since it's already a LabelFrame
        # with the title "Connection Options"
        self.options_frame = self.app.right_frame
        
        # Create a vertical layout for the connection options
        # Parameters section
        self.params_frame = ttk.LabelFrame(self.options_frame, text="Parameters")
        self.params_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        # Actions section
        self.actions_frame = ttk.LabelFrame(self.options_frame, text="Actions")
        self.actions_frame.pack(fill=tk.X, expand=False, padx=10, pady=10)
        
        # Add distance STD factor slider
        ttk.Label(self.params_frame, text="Distance STD Factor:").grid(
            column=0, row=0, padx=5, pady=5, sticky=tk.W
        )
        self.distance_std_factor_var = tk.DoubleVar(value=self.distance_std_factor)
        self.distance_std_factor_slider = ttk.Scale(
            self.params_frame, 
            from_=0.1, to=3.0, 
            orient=tk.HORIZONTAL,
            variable=self.distance_std_factor_var, 
            length=180, 
            command=self.update_distance_std_factor_display
        )
        self.distance_std_factor_slider.grid(column=0, row=1, padx=5, pady=5, sticky=tk.W+tk.E)
        self.distance_std_factor_display = ttk.Label(self.params_frame, text=str(self.distance_std_factor))
        self.distance_std_factor_display.grid(column=1, row=1, padx=5, pady=5, sticky=tk.W)
        
        # Add min connections factor slider
        ttk.Label(self.params_frame, text="Min Connections Factor:").grid(
            column=0, row=2, padx=5, pady=5, sticky=tk.W
        )
        self.min_connections_factor_var = tk.DoubleVar(value=self.min_connections_factor)
        self.min_connections_factor_slider = ttk.Scale(
            self.params_frame, 
            from_=0.1, to=3.0, 
            orient=tk.HORIZONTAL,
            variable=self.min_connections_factor_var, 
            length=180, 
            command=self.update_min_connections_factor_display
        )
        self.min_connections_factor_slider.grid(column=0, row=3, padx=5, pady=5, sticky=tk.W+tk.E)
        self.min_connections_factor_display = ttk.Label(self.params_frame, text=str(self.min_connections_factor))
        self.min_connections_factor_display.grid(column=1, row=3, padx=5, pady=5, sticky=tk.W)
        
        # Add direction STD factor slider
        ttk.Label(self.params_frame, text="Direction STD Factor:").grid(
            column=0, row=4, padx=5, pady=5, sticky=tk.W
        )
        self.direction_std_factor_var = tk.DoubleVar(value=self.direction_std_factor)
        self.direction_std_factor_slider = ttk.Scale(
            self.params_frame, 
            from_=0.1, to=3.0, 
            orient=tk.HORIZONTAL,
            variable=self.direction_std_factor_var, 
            length=180, 
            command=self.update_direction_std_factor_display
        )
        self.direction_std_factor_slider.grid(column=0, row=5, padx=5, pady=5, sticky=tk.W+tk.E)
        self.direction_std_factor_display = ttk.Label(self.params_frame, text=str(self.direction_std_factor))
        self.direction_std_factor_display.grid(column=1, row=5, padx=5, pady=5, sticky=tk.W)
        
        # Add NFD (Number of Direction Filtering Times) slider
        ttk.Label(self.params_frame, text="Direction Filter Iterations:").grid(
            column=0, row=6, padx=5, pady=5, sticky=tk.W
        )
        self.nfd_var = tk.IntVar(value=self.nfd)
        self.nfd_slider = ttk.Scale(
            self.params_frame, 
            from_=1, to=5, 
            orient=tk.HORIZONTAL,
            variable=self.nfd_var, 
            length=180, 
            command=self.update_nfd_display
        )
        self.nfd_slider.grid(column=0, row=7, padx=5, pady=5, sticky=tk.W+tk.E)
        self.nfd_display = ttk.Label(self.params_frame, text=str(self.nfd))
        self.nfd_display.grid(column=1, row=7, padx=5, pady=5, sticky=tk.W)
        
        # Show all connections button
        self.connections_button = ttk.Button(
            self.actions_frame, 
            text="Show All Connections", 
            command=self.show_all_connections
        )
        self.connections_button.pack(fill=tk.X, padx=10, pady=5)
        
        # Filter by count button
        self.count_filtered_button = ttk.Button(
            self.actions_frame, 
            text="Filter By Count", 
            command=self.show_count_filtered_connections
        )
        self.count_filtered_button.pack(fill=tk.X, padx=10, pady=5)
        
        # Filter by direction button
        self.direction_filtered_button = ttk.Button(
            self.actions_frame, 
            text="Filter By Direction", 
            command=self.show_direction_filtered_connections
        )
        self.direction_filtered_button.pack(fill=tk.X, padx=10, pady=5)
        
        # Plot cluster centers and k button
        self.centers_button = ttk.Button(
            self.actions_frame, 
            text="Plot Clusters Centers and k", 
            command=self.plot_cluster_centers
        )
        self.centers_button.pack(fill=tk.X, padx=10, pady=5)
    
    def update_distance_std_factor_display(self, event=None):
        """Update the display label for distance std factor"""
        self.distance_std_factor = round(float(self.distance_std_factor_var.get()), 2)
        self.distance_std_factor_display.config(text=f"{self.distance_std_factor:.2f}")
    
    def update_min_connections_factor_display(self, event=None):
        """Update the display label for min connections factor"""
        self.min_connections_factor = round(float(self.min_connections_factor_var.get()), 2)
        self.min_connections_factor_display.config(text=f"{self.min_connections_factor:.2f}")
    
    def update_direction_std_factor_display(self, event=None):
        """Update the display label for direction std factor"""
        self.direction_std_factor = round(float(self.direction_std_factor_var.get()), 2)
        self.direction_std_factor_display.config(text=f"{self.direction_std_factor:.2f}")
    
    def update_nfd_display(self, event=None):
        """Update the display label for number of direction filtering times"""
        self.nfd = int(self.nfd_var.get())
        self.nfd_display.config(text=str(self.nfd))
    
    def get_points_and_distances(self):
        """Get points and calculate distances"""
        # Check if we have enough points to process
        if len(self.app.points) < 2:
            self.app.status_var.set("Need at least 2 points to show connections")
            return None, None, None
        
        # Convert points to numpy array
        points = np.array(self.app.points)
        
        # Calculate distances between points
        distances, stats = calculate_point_distances(points)
        
        return points, distances, stats
    
    def show_all_connections(self):
        """Show all connections between points based on distance threshold"""
        # Get points and distances
        points, distances, stats = self.get_points_and_distances()
        if points is None:
            return
        
        # Get current parameters
        self.update_distance_std_factor_display()
        
        # Generate all connections using the distance std factor
        connections, threshold = generate_connections(
            points, 
            distances, 
            self.distance_std_factor
        )
        
        if not connections:
            self.app.status_var.set("No connections found with current distance threshold")
            return
        
        # Save the connections for possible use by other filters
        self.last_connections = connections
        self.last_points = points
        
        # Update status
        self.app.status_var.set(
            f"Found {len(connections)} connections with threshold {threshold:.2f}"
        )
        
        # Create visualization window
        create_connections_window(
            self.app.root,
            points,
            connections,
            threshold,
            self.app.img,
            "All Connections"
        )
    
    def show_count_filtered_connections(self):
        """Show connections after filtering out points with few connections"""
        # Get points and distances
        points, distances, stats = self.get_points_and_distances()
        if points is None:
            return
        
        # Get current parameters
        self.update_distance_std_factor_display()
        self.update_min_connections_factor_display()
        
        # Generate all connections using the distance std factor
        connections, threshold = generate_connections(
            points, 
            distances, 
            self.distance_std_factor
        )
        
        if not connections:
            self.app.status_var.set("No connections found with current distance threshold")
            return
        
        # Filter connections by count
        filtered_connections, min_required = filter_connections_by_count(
            connections,
            points,
            self.min_connections_factor
        )
        
        # Save the filtered connections for possible use by other filters
        self.last_connections = filtered_connections
        self.last_points = points
        
        # Update status
        self.app.status_var.set(
            f"Filtered from {len(connections)} to {len(filtered_connections)} connections. " +
            f"Min required connections: {min_required}"
        )
        
        # Create visualization window
        create_connections_window(
            self.app.root,
            points,
            filtered_connections,
            threshold,
            self.app.img,
            "Count-Filtered Connections",
            f"Min required connections: {min_required}"
        )
    
    def show_direction_filtered_connections(self):
        """
        Show connections after filtering based on their directional distribution.
        Uses the last generated connections (from either all connections or count filtered).
        Applies the direction filtering nfd times as specified by the user.
        """
        # Check if we have connections from a previous operation
        if self.last_connections is None or self.last_points is None:
            self.app.status_var.set("Please generate connections first using 'Show All Connections' or 'Filter By Count'")
            return
        
        # Get current parameters
        self.update_direction_std_factor_display()
        self.update_nfd_display()
        
        # Start with the last connections
        filtered_connections = self.last_connections
        total_connections_before = len(filtered_connections)
        
        # Apply directional filtering nfd times
        for i in range(self.nfd):
            # Apply the filter
            filtered_connections, info = filter_connections_by_direction(
                filtered_connections,
                self.last_points,
                self.direction_std_factor
            )
            
            # Check if we still have connections left
            if not filtered_connections:
                self.app.status_var.set(f"No connections left after {i+1} iterations of directional filtering")
                return
        
        # Save the filtered connections for possible further filtering
        self.last_connections = filtered_connections
        
        # Update status
        self.app.status_var.set(
            f"Filtered from {total_connections_before} to {len(filtered_connections)} connections " +
            f"after {self.nfd} iterations of directional filtering"
        )
        
        # Create visualization window
        create_connections_window(
            self.app.root,
            self.last_points,
            filtered_connections,
            0,  # Not using distance threshold
            self.app.img,
            "Direction-Filtered Connections",
            f"Direction STD factor: {self.direction_std_factor}, Iterations: {self.nfd}"
        )
    
    def plot_cluster_centers(self):
        """
        Plot cluster centers based on the last applied filter.
        Identifies connected components as separate clusters and calculates their centers.
        """
        # Check if we have connections from a previous operation
        if self.last_connections is None or self.last_points is None:
            self.app.status_var.set("Please generate connections first using one of the connection actions")
            return

        # Convert the connections to an adjacency list representation
        graph = {}
        for i, j in self.last_connections:
            if i not in graph:
                graph[i] = []
            if j not in graph:
                graph[j] = []
            graph[i].append(j)
            graph[j].append(i)
        
        # Initialize visited set for DFS
        visited = set()
        clusters = []
        
        # DFS function to find connected components (clusters)
        def dfs(node, cluster):
            visited.add(node)
            cluster.append(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, cluster)
        
        # Find all connected components (clusters)
        for node in range(len(self.last_points)):
            if node in graph and node not in visited:
                cluster = []
                dfs(node, cluster)
                if cluster:  # Only add non-empty clusters
                    clusters.append(cluster)
        
        # Calculate the center of each cluster (mean of all points in the cluster)
        cluster_centers = []
        for cluster in clusters:
            cluster_points = self.last_points[cluster]
            center = np.mean(cluster_points, axis=0)
            cluster_centers.append(center)
        
        # Create a new window to display the clusters and centers
        centers_window = tk.Toplevel(self.app.root)
        centers_window.title(f"Cluster Centers (k={len(clusters)})")
        centers_window.geometry("800x700")
        
        # Create a matplotlib figure
        fig = Figure(figsize=(8, 7), dpi=100)
        ax = fig.add_subplot(111)
        
        # Plot all points
        ax.scatter(self.last_points[:, 0], self.last_points[:, 1], s=20, c='blue', alpha=0.5)
        
        # Plot the connections
        for i, j in self.last_connections:
            ax.plot([self.last_points[i, 0], self.last_points[j, 0]], 
                    [self.last_points[i, 1], self.last_points[j, 1]], 'k-', alpha=0.3)
        
        # Different colors for different clusters
        colors = plt.cm.tab10.colors
        
        # Plot each cluster with a different color
        for i, cluster in enumerate(clusters):
            cluster_color = colors[i % len(colors)]
            cluster_points = self.last_points[cluster]
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       s=30, c=[cluster_color], alpha=0.7, label=f'Cluster {i+1}')
        
        # Plot the centers
        centers_array = np.array(cluster_centers)
        ax.scatter(centers_array[:, 0], centers_array[:, 1], 
                   s=200, c='red', marker='X', edgecolors='black', label='Cluster Centers')
        
        ax.set_title(f'Identified Clusters: k={len(clusters)}')
        ax.legend()
        ax.grid(True)
        
        # Create an info frame for details
        info_frame = ttk.Frame(centers_window)
        info_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # Add information about the clusters
        info_text = f"Number of clusters (k): {len(clusters)}\n"
        info_text += f"Cluster sizes: {', '.join([str(len(c)) for c in clusters])}\n"
        info_text += "Cluster centers (x, y):\n"
        for i, center in enumerate(cluster_centers):
            info_text += f"  Cluster {i+1}: ({center[0]:.1f}, {center[1]:.1f})\n"
        
        info_label = ttk.Label(info_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(side=tk.LEFT, padx=10)
        
        # Add a close button
        close_button = ttk.Button(info_frame, text="Close", command=centers_window.destroy)
        close_button.pack(side=tk.RIGHT, padx=10)
        
        # Bind escape key to close the window
        centers_window.bind("<Escape>", lambda e: centers_window.destroy())
        
        # Embed the matplotlib figure in the window
        canvas = FigureCanvasTkAgg(fig, master=centers_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Update status
        self.app.status_var.set(f"Identified {len(clusters)} clusters from the connections")

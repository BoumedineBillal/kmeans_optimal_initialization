import os
import sys
import numpy as np

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from version2.utils.distance_utils import (
    calculate_point_distances,
    generate_connections,
    filter_connections_by_count,
    filter_connections_by_direction
)

# Parameters - SAME AS IN THE GUI
DISTANCE_STD_FACTOR = 0.71
MIN_CONNECTIONS_FACTOR = 1.0
DIRECTION_STD_FACTOR = 2.0
NFD = 2

# Path to the points directory
POINTS_DIR = r"C:\Users\orani\bilel\a_miv\a_miv\m1s2\fd\rapport2\version2\points"

def load_points(file_path):
    return np.load(file_path)

def identify_clusters(points_file):
    """
    Identify clusters using EXACTLY the same algorithm as in the GUI's
    plot_cluster_centers method in ConnectionsExtension class.
    """
    print(f"Processing {points_file}...")
    points = load_points(points_file)
    
    # Apply the sequence of filters
    distances, _ = calculate_point_distances(points)
    connections, _ = generate_connections(points, distances, DISTANCE_STD_FACTOR)
    count_filtered, _ = filter_connections_by_count(connections, points, MIN_CONNECTIONS_FACTOR)
    
    current_connections = count_filtered
    for _ in range(NFD):
        if current_connections:
            current_connections, _ = filter_connections_by_direction(
                current_connections, points, DIRECTION_STD_FACTOR
            )
    
    # Convert the connections to an adjacency list representation
    graph = {}
    for i, j in current_connections:
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
    for node in range(len(points)):
        if node in graph and node not in visited:
            cluster = []
            dfs(node, cluster)
            if cluster:  # Only add non-empty clusters
                clusters.append(cluster)
    
    # Calculate the center of each cluster (mean of all points in the cluster)
    cluster_centers = []
    for cluster in clusters:
        cluster_points = points[cluster]
        center = np.mean(cluster_points, axis=0)
        cluster_centers.append(center)
    
    # Output the results
    print(f"Number of clusters (k): {len(clusters)}")
    print(f"Cluster sizes: {', '.join([str(len(c)) for c in clusters])}")
    print("Cluster centers (x, y):")
    for i, center in enumerate(cluster_centers):
        print(f"  Cluster {i+1}: ({center[0]:.1f}, {center[1]:.1f})")
    
    print("\n" + "-"*50 + "\n")
    return len(clusters), np.array(cluster_centers) if cluster_centers else None

def main():
    """Process all point sets to verify k values and centers"""
    # Process each points file
    for filename in os.listdir(POINTS_DIR):
        if filename.endswith('.npy'):
            points_file = os.path.join(POINTS_DIR, filename)
            k, centers = identify_clusters(points_file)
            print(f"File: {filename}")
            print(f"k = {k}")
            if centers is not None:
                for i, center in enumerate(centers):
                    print(f"Center {i+1}: ({center[0]:.1f}, {center[1]:.1f})")
            print()

if __name__ == "__main__":
    main()

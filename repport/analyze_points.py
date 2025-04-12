import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from version2.utils.distance_utils import (
    calculate_point_distances,
    generate_connections,
    filter_connections_by_count,
    filter_connections_by_direction
)

# Parameters
DISTANCE_STD_FACTOR = 0.71
MIN_CONNECTIONS_FACTOR = 1.0
DIRECTION_STD_FACTOR = 2.0
NFD = 2

# Path to the points directory
POINTS_DIR = r"C:\Users\orani\bilel\a_miv\a_miv\m1s2\fd\rapport2\version2\points"
REPORT_DIR = r"C:\Users\orani\bilel\a_miv\a_miv\m1s2\fd\rapport2\version2\repport"

# Create plots directory if it doesn't exist
PLOTS_DIR = os.path.join(REPORT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_points(file_path):
    return np.load(file_path)

def calculate_elbow_scores(points, max_k=10):
    """
    Calculate inertia (within-cluster sum of squares) for different k values
    """
    inertias = []
    silhouette_scores = []
    
    # Calculate for k from 2 to max_k
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(points)
        inertias.append(kmeans.inertia_)
        
        # Only calculate silhouette score if k > 1 and there are at least k+1 samples
        if k > 1 and len(points) > k:
            try:
                score = silhouette_score(points, kmeans.labels_)
                silhouette_scores.append(score)
            except:
                silhouette_scores.append(0)
    
    return inertias, silhouette_scores

def plot_elbow_method(points_file, save_dir=PLOTS_DIR):
    """
    Plot the Elbow method graph comparing traditional elbow method with our method
    to determine optimal k value
    """
    points = load_points(points_file)
    
    # Create figure for elbow method comparison
    plt.figure(figsize=(8, 6))
    
    # Get data points and calculate elbow curve
    points = load_points(points_file)
    inertias, silhouette_scores = calculate_elbow_scores(points)
    k_range = range(2, len(inertias)+2)
    
    # Get our method's k determination
    our_k, initial_centers = identify_optimal_k(points_file)
    
    # Plot the elbow curve
    plt.plot(k_range, inertias, 'o-', color='blue', linewidth=2, label='Inertie')
    
    # Mark our method's k
    our_k_inertia = inertias[our_k-2] if our_k-2 < len(inertias) else inertias[-1]
    plt.scatter(our_k, our_k_inertia, color='red', s=200, marker='X', linewidth=3, label=f'Notre méthode (k={our_k})')
    
    # Add labels
    plt.title('Détermination du nombre optimal de clusters (k)')
    plt.xlabel('Nombre de clusters (k)')
    plt.ylabel('Inertie')
    plt.grid(True)
    plt.legend()
    plt.xticks(k_range)
    
    plt.tight_layout()
    
    # Save the elbow method figure
    base_filename = os.path.basename(points_file).replace('.npy', '')
    elbow_fig_path = os.path.join(save_dir, f"{base_filename}_elbow.png")
    plt.savefig(elbow_fig_path)
    plt.close()
    
    return elbow_fig_path

def identify_optimal_k(points_file):
    """
    Identify the optimal k and centers using the exact same method as the GUI.
    Uses DFS to find connected components after filtering.
    """
    points = load_points(points_file)
    
    # Apply the sequence of filters to identify potential cluster centers
    distances, _ = calculate_point_distances(points)
    connections, _ = generate_connections(points, distances, DISTANCE_STD_FACTOR)
    count_filtered, _ = filter_connections_by_count(connections, points, MIN_CONNECTIONS_FACTOR)
    
    current_connections = count_filtered
    for _ in range(NFD):
        if current_connections:
            current_connections, _ = filter_connections_by_direction(
                current_connections, points, DIRECTION_STD_FACTOR
            )
    
    # Use exactly the same approach as in the GUI's plot_cluster_centers method
    if current_connections:
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
        
        # The k is the number of connected components found
        k = len(clusters)
        if k >= 2:  # Make sure we have at least 2 clusters
            return k, np.array(cluster_centers)
    
    # Default to k=3 if our method fails
    print("Warning: Could not determine k and centers, defaulting to k=3")
    return 3, None

def compare_convergence_time(points_file, save_dir=PLOTS_DIR):
    """
    Compare the convergence time between:
    1. Standard K-means with random initialization (using our k value)
    2. K-means using our determined centers (and our k value)
    """
    # Identify k and initial centers using the same method as the GUI
    points = load_points(points_file)  # Load points first
    optimal_k, initial_centers = identify_optimal_k(points_file)  # Get k and centers using DFS
    
    # Save the initial random centers used by standard k-means for visualization
    np.random.seed(42)  # Make sure we use the same seed as KMeans
    random_idx = np.random.choice(len(points), optimal_k, replace=False)
    random_init = points[random_idx]
    
    # Measure time for standard k-means
    start_time = time.time()
    standard_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    standard_kmeans.fit(points)
    standard_time = time.time() - start_time
    # Store initial centers for visualization
    standard_kmeans.init_centers_ = random_init
    
    # Measure time for k-means with our centers
    our_time = None
    if initial_centers is not None:
        start_time = time.time()
        our_kmeans = KMeans(n_clusters=optimal_k, init=initial_centers, n_init=1)
        our_kmeans.fit(points)
        our_time = time.time() - start_time
    
    # Create comparison bar chart
    plt.figure(figsize=(6, 5))
    methods = ["K-means Standard"]
    times = [standard_time]
    
    if our_time is not None:
        methods.append("Notre Méthode")
        times.append(our_time)
    
    plt.bar(methods, times, color=['lightblue', 'orange'])
    plt.title(f'Comparaison des Temps de Convergence pour k={optimal_k}')
    plt.xlabel('Méthode')
    plt.ylabel('Temps (secondes)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(times):
        plt.text(i, v + 0.001, f"{v:.4f}s", ha='center')
    
    # Save the convergence time figure
    base_filename = os.path.basename(points_file).replace('.npy', '')
    conv_fig_path = os.path.join(save_dir, f"{base_filename}_convergence.png")
    plt.savefig(conv_fig_path)
    plt.close()
    
    # Also create a figure showing the clustering results
    plt.figure(figsize=(7, 6))
    
    # Plot our method's k-means result if available
    if our_time is not None:
        plt.scatter(points[:, 0], points[:, 1], c=our_kmeans.labels_, cmap='viridis', s=25, alpha=0.7)
        plt.scatter(our_kmeans.cluster_centers_[:, 0], our_kmeans.cluster_centers_[:, 1], 
                    c='red', marker='X', s=100, label='Centres finaux')
        # Plot our initial centers
        plt.scatter(initial_centers[:, 0], initial_centers[:, 1],
                   c='green', marker='o', s=80, alpha=0.5, label='Centres initiaux (notre méthode)')
        plt.title(f'K-means avec notre initialisation (k={optimal_k})')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    # Save the clustering figure
    cluster_fig_path = os.path.join(save_dir, f"{base_filename}_clusters.png")
    plt.savefig(cluster_fig_path)
    plt.close()
    
    return conv_fig_path, cluster_fig_path, optimal_k, standard_time, our_time

def visualize_connections(points_file, save_dir=PLOTS_DIR):
    """
    Visualize the points and connections at each stage of filtering
    """
    points = load_points(points_file)
    
    # Create a 2x2 grid of plots
    plt.figure(figsize=(9, 7))
    
    # Original points
    plt.subplot(2, 2, 1)
    plt.scatter(points[:, 0], points[:, 1], s=20, c='blue', alpha=0.7)
    plt.title('Points d\'origine')
    plt.grid(True)
    
    # After all connections filter
    distances, _ = calculate_point_distances(points)
    connections, _ = generate_connections(points, distances, DISTANCE_STD_FACTOR)
    
    plt.subplot(2, 2, 2)
    plt.scatter(points[:, 0], points[:, 1], s=20, c='blue', alpha=0.7)
    
    # Draw connections
    for i, j in connections:
        plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'k-', alpha=0.2)
    
    plt.title('Après "Show All Connections"')
    plt.grid(True)
    
    # After count filter
    count_filtered, _ = filter_connections_by_count(connections, points, MIN_CONNECTIONS_FACTOR)
    
    plt.subplot(2, 2, 3)
    plt.scatter(points[:, 0], points[:, 1], s=20, c='blue', alpha=0.7)
    
    # Draw connections
    for i, j in count_filtered:
        plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'k-', alpha=0.3)
    
    plt.title('Après "Filter By Count"')
    plt.grid(True)
    
    # After direction filter (applied NFD times)
    current_connections = count_filtered
    for _ in range(NFD):
        if current_connections:
            current_connections, _ = filter_connections_by_direction(
                current_connections, points, DIRECTION_STD_FACTOR
            )
    
    plt.subplot(2, 2, 4)
    plt.scatter(points[:, 0], points[:, 1], s=20, c='blue', alpha=0.7)
    
    # Draw connections
    for i, j in current_connections:
        plt.plot([points[i, 0], points[j, 0]], [points[i, 1], points[j, 1]], 'k-', alpha=0.5)
    
    # Highlight potential cluster centers
    if current_connections:
        # Count connections per point
        point_connections = {}
        for i, j in current_connections:
            if i not in point_connections:
                point_connections[i] = []
            if j not in point_connections:
                point_connections[j] = []
            point_connections[i].append(j)
            point_connections[j].append(i)
        
        point_scores = np.zeros(len(points))
        for idx, connected_points in point_connections.items():
            point_scores[idx] = len(connected_points)
        
        if point_scores.max() > 0:
            normalized_scores = point_scores / point_scores.max()
            potential_centers_idx = np.where(normalized_scores > 0.5)[0]
            
            # Highlight the potential centers
            plt.scatter(points[potential_centers_idx, 0], points[potential_centers_idx, 1], 
                        s=100, c='red', marker='X', label='Centres potentiels')
            plt.legend()
    
    plt.title('Après "Filter By Direction"')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save the connections visualization
    base_filename = os.path.basename(points_file).replace('.npy', '')
    connections_fig_path = os.path.join(save_dir, f"{base_filename}_connections.png")
    plt.savefig(connections_fig_path)
    plt.close()
    
    return connections_fig_path

def main():
    """Process all point sets and generate all figures for the report"""
    results = []
    
    # Process each points file
    for filename in os.listdir(POINTS_DIR):
        if filename.endswith('.npy'):
            points_file = os.path.join(POINTS_DIR, filename)
            print(f"Processing {filename}...")
            
            # Identify the connected components automatically
            # Using EXACTLY the same method as in the GUI
            optimal_k, initial_centers = identify_optimal_k(points_file)
            print(f"Found {optimal_k} connected components (clusters) with centers at:")
            for i, center in enumerate(initial_centers):
                print(f"  Cluster {i+1}: ({center[0]:.1f}, {center[1]:.1f})")
                
            # Generate elbow method plots
            elbow_path = plot_elbow_method(points_file)
            
            # Generate convergence time comparison
            conv_path, cluster_path, optimal_k, std_time, our_time = compare_convergence_time(points_file)
            
            # Generate connections visualization
            connections_path = visualize_connections(points_file)
            
            # Save the results
            result = {
                'filename': filename,
                'optimal_k': optimal_k,
                'standard_time': std_time,
                'our_time': our_time,
                'elbow_path': elbow_path,
                'convergence_path': conv_path,
                'cluster_path': cluster_path,
                'connections_path': connections_path
            }
            results.append(result)
    
    # Print summary
    print("\nSummary of Results:")
    print("-------------------")
    for result in results:
        print(f"File: {result['filename']}")
        print(f"  Optimal k: {result['optimal_k']}")
        print(f"  Standard KMeans time: {result['standard_time']:.4f}s")
        if result['our_time']:
            print(f"  Our method time: {result['our_time']:.4f}s")
            improvement = (result['standard_time'] - result['our_time']) / result['standard_time'] * 100
            print(f"  Improvement: {improvement:.2f}%")
        else:
            print("  Our method: Failed to identify centers")
        print()
    
    # Save results to a JSON file for the update_report.py script
    results_path = os.path.join(REPORT_DIR, "analysis_results.json")
    
    # Convert numpy floats to Python floats for JSON serialization
    json_results = []
    for result in results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.float32) or isinstance(value, np.float64):
                json_result[key] = float(value)
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()

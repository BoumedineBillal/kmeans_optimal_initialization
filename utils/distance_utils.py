import numpy as np

def calculate_point_distances(points):
    """
    Calculate pairwise distances between all points in the dataset.
    
    Args:
        points: Array of point coordinates
        
    Returns:
        distances: Matrix of distances between points
        stats: Dictionary with statistics (min, max, mean, median, std)
    """
    if points is None or len(points) == 0:
        return None, None
    
    # Calculate pairwise distances between all points
    num_points = len(points)
    distances = np.zeros((num_points, num_points))
    
    for i in range(num_points):
        for j in range(i+1, num_points):
            # Calculate Euclidean distance
            dist = np.sqrt(np.sum((points[i] - points[j])**2))
            # Store distance in both positions of the symmetric matrix
            distances[i, j] = dist
            distances[j, i] = dist
    
    # Calculate statistics from upper triangular part (without diagonal)
    flat_distances = distances[np.triu_indices(num_points, k=1)]
    stats = {
        "min": np.min(flat_distances) if len(flat_distances) > 0 else 0,
        "max": np.max(flat_distances) if len(flat_distances) > 0 else 0,
        "mean": np.mean(flat_distances) if len(flat_distances) > 0 else 0,
        "median": np.median(flat_distances) if len(flat_distances) > 0 else 0,
        "std": np.std(flat_distances) if len(flat_distances) > 0 else 0
    }
    
    return distances, stats

def generate_connections(points, distances, std_factor=1.0):
    """
    Generate a list of pairs of points that are closer than std_factor * standard deviation.
    
    Args:
        points: Array of point coordinates
        distances: Matrix of distances between points
        std_factor: Multiplier for the standard deviation threshold (default: 1.0)
        
    Returns:
        List of tuples: Each tuple contains indices of two points that are close to each other
        threshold: The distance threshold used
    """
    if distances is None:
        return [], 0
    
    # Calculate stats directly if not provided
    flat_distances = distances[np.triu_indices(len(points), k=1)]
    std = np.std(flat_distances) if len(flat_distances) > 0 else 0
    
    threshold = std * std_factor
    num_points = len(points)
    
    # Find pairs of points with distances less than the threshold
    connections = []
    for i in range(num_points):
        for j in range(i+1, num_points):
            if distances[i, j] < threshold:
                connections.append((i, j))
    
    return connections, threshold

def filter_connections_by_count(connections, points, min_connections_factor):
    """
    Filter connections to only include points that have enough connections.
    
    Args:
        connections: List of tuples with point indices
        points: Array of point coordinates
        min_connections_factor: Factor to determine minimum required connections
        
    Returns:
        List of filtered connections
        min_required: Minimum number of connections required
    """
    if not connections:
        return [], 0
    
    # Count connections per point
    point_connection_counts = [0] * len(points)
    for i, j in connections:
        point_connection_counts[i] += 1
        point_connection_counts[j] += 1
    
    # Calculate the standard deviation and mean of connections per point
    connections_std = np.std(point_connection_counts)
    connections_mean = np.mean(point_connection_counts)
    
    # Minimum required connections based on the factor
    min_required = max(1, int(connections_mean - (connections_std * min_connections_factor)))
    
    # Filter the connections
    filtered_connections = []
    for i, j in connections:
        if (point_connection_counts[i] >= min_required and 
            point_connection_counts[j] >= min_required):
            filtered_connections.append((i, j))
    
    return filtered_connections, min_required

def circular_gaussian_filter(vec, sigma=5):
    """
    Apply a Gaussian filter to a circular vector, handling the wrap-around at indices 0 and 359.
    
    Args:
        vec: A 1D array representing the circular vector (e.g., 360 degrees)
        sigma: Standard deviation for the Gaussian kernel
        
    Returns:
        Smoothed circular vector
    """
    n = len(vec)
    # Create a larger vector with padding on both sides to handle the circular boundary
    padding = 3 * sigma
    padded_vec = np.zeros(n + 2 * padding)
    
    # Fill the padded vector with the original values and the wrapped values
    padded_vec[padding:padding+n] = vec
    padded_vec[:padding] = vec[-(padding):]  # Wrap end to beginning
    padded_vec[-(padding):] = vec[:padding]  # Wrap beginning to end
    
    # Apply Gaussian smoothing
    # Create a Gaussian kernel
    x = np.arange(-3*sigma, 3*sigma+1)
    kernel = np.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / np.sum(kernel)  # Normalize
    
    # Apply convolution
    result = np.zeros_like(padded_vec)
    for i in range(padding, padding + n):
        # Apply kernel centered at position i
        window = padded_vec[i-len(kernel)//2:i+len(kernel)//2+1]
        if len(window) == len(kernel):
            result[i] = np.sum(window * kernel)
        else:
            # Handle edge cases
            result[i] = padded_vec[i]
    
    # Extract the relevant part
    return result[padding:padding+n]

def filter_connections_by_direction(connections, points, std_factor):
    """
    Filter connections based on their directional distribution around each point.
    For each point, analyze the directions of its connections and keep only those
    in significant directions (above a threshold determined by std * factor).
    
    Args:
        connections: List of tuples with indices of points to connect
        points: Array of point coordinates
        std_factor: Factor to multiply by the std deviation for thresholding
        
    Returns:
        List of filtered connections
        Information about the filtering process
    """
    if not connections:
        return [], {"threshold": 0, "total_removed": 0}
    
    # Create a dictionary to store connections per point
    point_connections = {}
    for i, j in connections:
        if i not in point_connections:
            point_connections[i] = []
        if j not in point_connections:
            point_connections[j] = []
        point_connections[i].append(j)
        point_connections[j].append(i)
    
    # Track which connections to keep
    connections_to_keep = set()
    
    # Process each point and its connections
    for point_idx, connected_points in point_connections.items():
        # Skip points with too few connections
        if len(connected_points) < 3:  # Need at least 3 for meaningful directional analysis
            for other_idx in connected_points:
                if point_idx < other_idx:  # Avoid duplicates
                    connections_to_keep.add((point_idx, other_idx))
                else:
                    connections_to_keep.add((other_idx, point_idx))
            continue
        
        # Calculate angles for each connection
        angles = []
        for other_idx in connected_points:
            dx = points[other_idx][0] - points[point_idx][0]
            dy = points[other_idx][1] - points[point_idx][1]
            angle = int(np.degrees(np.arctan2(dy, dx))) % 360
            angles.append((other_idx, angle))
        
        # Create the circular vector of angles
        vec_angles = np.zeros(360)
        for _, angle in angles:
            vec_angles[angle] += 1
        
        # Apply circular Gaussian filter
        vec_c = circular_gaussian_filter(vec_angles)
        
        # Calculate standard deviation and threshold
        std_vec_c = np.std(vec_c)
        threshold = std_vec_c * std_factor
        
        # Filter connections based on the threshold
        for other_idx, angle in angles:
            if vec_c[angle] >= threshold:
                # Keep this connection
                if point_idx < other_idx:  # Avoid duplicates
                    connections_to_keep.add((point_idx, other_idx))
                else:
                    connections_to_keep.add((other_idx, point_idx))
    
    # Convert set to list
    filtered_connections = list(connections_to_keep)
    
    # Prepare information about the filtering
    info = {
        "threshold": threshold if 'threshold' in locals() else 0,
        "total_removed": len(connections) - len(filtered_connections)
    }
    
    return filtered_connections, info
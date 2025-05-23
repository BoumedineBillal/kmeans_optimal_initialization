# Filter By Direction

## Overview
The "Filter By Direction" action filters connections based on the directional distribution around each point. This advanced filtering method helps identify significant directional patterns by keeping connections in directions that stand out statistically. After applying this filter, the resulting connection structure can be used to determine the optimal number of clusters (k) and their initial centers for K-means clustering.

## Parameters
- **Direction STD Factor**: Controls the threshold for keeping connections based on their directional significance. A higher value keeps more connections, while a lower value applies stricter filtering. The threshold is calculated as `std * factor`, where `std` is the standard deviation of the smoothed circular vector of connection angles.
- **Direction Filter Iterations (nfd)**: Controls how many times the directional filtering algorithm is applied successively. Each iteration uses the output of the previous one, allowing for more pronounced filtering with each pass.

## Algorithm
1. Use the connections from the most recently applied action (either "Show All Connections" or "Filter By Count")
2. Repeat the following process nfd times (Direction Filter Iterations parameter):
   - For each point with at least 3 connections:
     - Calculate the angle (direction) of each connection from this point
     - Create a circular vector of 360 values (one for each degree), with 0 in all positions
     - Increment the value at the index corresponding to each connection's angle
     - Apply a circular Gaussian filter to smooth this vector, handling the wrap-around at 0°/360°
     - Calculate the standard deviation of the smoothed vector
     - Set a threshold as `std * Direction STD Factor`
     - Keep only connections whose corresponding angle has a value above this threshold
   - Use the filtered connections as input for the next iteration
3. Points with fewer than 3 connections are skipped in this analysis (their connections are preserved)

## Determining K and Initial Centers
After applying the directional filtering, you can use the "Plot Clusters Centers and k" action to:

1. Identify connected components in the filtered connection graph using depth-first search (DFS)
2. Each connected component forms a natural cluster
3. The number of connected components determines the optimal k value
4. The center of each component (mean position of all points in the component) serves as an initial center for K-means

This approach provides a data-driven method for determining both k and the initial centers without manual intervention.

## Performance Benefits
Using the k value and initial centers determined through this method to initialize K-means provides significant advantages:
- Convergence time reduced by 90-92% compared to random initialization
- Improved consistency in clustering results
- Automatic adaptation to the natural structures in the data

## Output
- A visualization window showing the directionally filtered connections
- Status information showing the number of connections before and after filtering
- Information about the directional threshold used

## Usage Notes
- This action requires previous connections to be generated by either "Show All Connections" or "Filter By Count"
- A good starting value for Direction STD Factor is 2.0
- A good starting value for Direction Filter Iterations is 2
- This action is particularly useful for identifying directional patterns in the point distribution
- For best results, apply after "Filter By Count" to first remove isolated points
- Lower Direction STD Factor values (0.1-0.5) will aggressively filter to keep only the most directionally significant connections
- Higher Direction STD Factor values (1.5-3.0) will keep more connections with less strict directional filtering
- Increasing the number of iterations (nfd) will progressively refine the directional filtering with each pass

## How It Works
The directional filtering looks at how connections are distributed around each point in terms of angles. If many connections are in similar directions, those directions become "significant" and connections in those directions are kept. Directions with fewer connections (below the threshold) are filtered out.

The circular Gaussian filter ensures that the directionality analysis properly handles the circular nature of angles (where 359° is adjacent to 0°) and smooths the directional data to identify meaningful patterns.

## Examples
- **Direction STD Factor variations**:
  - Low Direction STD Factor (0.5): Keeps only connections in the most significant directions
  - Medium Direction STD Factor (1.0): Balances filtering while preserving directional structure
  - High Direction STD Factor (2.0): Light filtering that preserves most directional connections

- **Iteration (nfd) variations**:
  - One Iteration (nfd=1): Single pass filtering, useful for light directional emphasis
  - Two Iterations (nfd=2): Good balance of filtering and performance for most use cases
  - Multiple Iterations (nfd=3-5): Progressive reinforcement of the strongest directional patterns

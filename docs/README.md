# K-means Clustering Tool v2 Documentation

## Overview
This application provides interactive K-means clustering functionality with advanced point connection analysis. It allows you to generate random points, run K-means clustering, and analyze the connections between points using various filtering methods. The connection analysis can automatically determine the optimal number of clusters (k) and their initial centers for K-means, significantly improving convergence performance.

## Main Features
1. **Point Generation**: Create random point distributions by clicking on the canvas
2. **K-means Clustering**: Run clustering analysis with customizable parameters
3. **Connection Analysis**: Generate and filter connections between points based on various criteria
4. **Automatic K Determination**: Identify the optimal number of clusters (k) through connection analysis
5. **Optimal Centers Initialization**: Determine initial centers for K-means to improve convergence

## Connection Analysis Actions
The application provides four connection analysis actions that can be used sequentially:

1. **[Show All Connections](all_connections.md)**: Generate the initial set of connections based on distance threshold
2. **[Filter By Count](count_filtered_connections.md)**: Filter connections by removing points with too few connections
3. **[Filter By Direction](direction_filtered_connections.md)**: Filter connections based on their directional distribution
4. **[Plot Clusters Centers and k](plot_clusters_centers.md)**: Identify connected components as clusters and determine their centers

## Typical Workflow
1. Generate points by clicking on the canvas
2. Generate all connections using "Show All Connections"
3. Apply count filtering using "Filter By Count" 
4. Apply directional filtering using "Filter By Direction"
5. View the identified cluster centers and k value using "Plot Clusters Centers and k"
6. Run K-means with the automatically determined k and centers

Each filtering step refines the connections, revealing different aspects of the point distribution structure.

## Parameters
Each action has specific parameters that control its behavior:

- **Distance STD Factor**: Controls the distance threshold for generating connections
- **Min Connections Factor**: Controls the minimum number of connections required in count filtering
- **Direction STD Factor**: Controls the directional significance threshold in directional filtering

Refer to the individual action documentation for detailed explanations of these parameters.

## Performance Benefits
When using the automatically determined k and centers to initialize K-means:
- Convergence time is reduced by 90-92% compared to standard random initialization
- The quality of clustering results is maintained or improved
- The algorithm automatically adapts to the natural structures in the data

## Tips for Effective Analysis
- Start with a Distance STD Factor around 1.0 to generate a reasonable number of initial connections
- Try different combinations of parameter settings to reveal different patterns
- The directional filtering is most effective when applied after count filtering
- Generate a larger number of points (100+) to see meaningful patterns in the connection analysis
- Experiment with different point distributions to see how the filters behave

## Additional Resources
- [Edit the `docs/` Folder on GitHub](https://github.com/BoumedineBillal/kmeans_optimal_initialization/edit/master/docs)

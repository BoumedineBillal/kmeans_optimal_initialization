# Filter By Count

## Overview
The "Filter By Count" action filters connections by removing those connected to points that have too few total connections. This helps identify more significant connection clusters by removing isolated points or those with minimal connectivity.

## Parameters
- **Distance STD Factor**: Controls the initial connections generation (inherited from "Show All Connections")
- **Min Connections Factor**: Controls the threshold for the minimum number of connections a point must have. A lower value keeps more connections, a higher value enforces stricter filtering. The threshold is calculated as `mean - (std * factor)`, where `mean` and `std` are the mean and standard deviation of the number of connections per point.

## Algorithm
1. First generate all connections using the Distance STD Factor (same as "Show All Connections")
2. Count the number of connections for each point
3. Calculate the mean and standard deviation of connection counts per point
4. Set a minimum required connections threshold as `mean - (std * Min Connections Factor)`
5. Filter out connections where either connecting point has fewer than the minimum required connections

## Output
- A visualization window showing filtered connections
- Status information showing the number of connections before and after filtering
- Information about the minimum required connections threshold used

## Usage Notes
- This action is typically used after "Show All Connections"
- A good starting value for Min Connections Factor is 1.0
- Decreasing the factor (closer to 0.1) will keep more connections
- Increasing the factor (toward 3.0) will aggressively filter connections
- This action saves the filtered connections for potential use by subsequent filtering actions

## Examples
- Low Min Connections Factor (0.5): Keeps most connections, filtering only very isolated points
- Medium Min Connections Factor (1.0): Moderately filters connections, keeping points with average connectivity
- High Min Connections Factor (2.0): Aggressively filters connections, keeping only points with many connections

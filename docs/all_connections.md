# Show All Connections

## Overview
The "Show All Connections" action generates connections between points based on their spatial proximity. This is the base action that creates the initial set of connections used by other filtering methods.

## Parameters
- **Distance STD Factor**: Controls the threshold for connecting points. A higher value creates more connections by including points that are farther apart. The threshold is calculated as `std * factor`, where `std` is the standard deviation of all point-to-point distances.

## Algorithm
1. Calculate all pairwise distances between points
2. Compute the standard deviation (std) of all distances
3. Set a threshold as `std * Distance STD Factor`
4. Create connections between all pairs of points with distances below this threshold

## Output
- A visualization window showing all points with lines connecting those that meet the threshold criteria
- Status information showing the number of connections found and the threshold value used

## Usage Notes
- This is usually the first step in the connection analysis workflow
- A good starting value for Distance STD Factor is 1.0
- Increasing the factor will result in more connections
- This action saves the generated connections for potential use by subsequent filtering actions

## Examples
- Low Distance STD Factor (0.5): Creates sparse connections between very close points
- Medium Distance STD Factor (1.0): Creates a moderate number of connections
- High Distance STD Factor (2.0): Creates many connections including between distant points

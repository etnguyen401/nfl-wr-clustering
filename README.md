# nfl-wr-clustering
This project categorizes wide receivers into four different clusters based on NFL Combine data using PCA, and k-means, and then creates graphs to visualize
the clusters representing different types of receivers.

## What it does
- Gets combine data for wide receivers and cleans it
- Imputes missing values
- Applies PCA to reduce the feature space
- Clusters players with k-means
- Generates interactive HTML plots for the cluster results

## Requirements
- Python 3.10+
- Dependencies listed in pyproject.toml

## Outputs
The script writes results into the data directory, including:
- CSV files with cleaned/imputed data
- PCA rotation and explained variance files
- Interactive HTML plots for the cluster visualizations

## Notes
The pipeline is implemented in the main module under the nfl-wr-k-means-clustering package.

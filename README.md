# nfl-wr-clustering
This project categorizes wide receivers into four different clusters based on NFL Combine data using PCA, and k-means, and then creates graphs to visualize
the clusters representing different types of receivers.

## What it does
- Gets combine data for wide receivers and cleans it
- Imputes missing values
- Applies PCA to reduce the feature space
- Clusters players with k-means
- Generates interactive HTML plots for the cluster results

## Mini Report
Firstly, I got the combine data from all of the players from nflreadpy. The columns that will be used to group wide receivers together into clusters will be measurables from the NFL Combine: height, weight, forty time, vertical, bench, broad jump, cone, and shuttle. 

A summary of how these measurables affect a wide receiver:
Height allows for receivers to make plays on the ball in the air and in contested situations when fighting for position with the cornerback.
Weight helps with physicality at the catch point, not being able to be pushed off of your route, and helps with staying healthy and durability.
Bench helps with upper body strength, beating press coverage, and blocking. 
Broad Jump measures lower body explosiveness and is a good test of athleticism.
The three cone measures how quickly a player can change direction, bend, and accelerate. 
The shuttle evaluates lateral quickness, and being able to stop on a dime and burst out of the stop.

Afterwards, I converted the height column from ft and inches to just inches, and kept the old column for easier understanding.

Then, the data is imputed so that all of the missing drills or measurements that the players are missing are filled to give a statistical estimate. This is done using either k-nearest neighbours or other methods. In this case, I used iterative imputation and Random Forest Regressor to find patterns along the features and uses them to fill in the missing values.

Subsequently, because there is so much overlapping variables to take in from the players we then perform PCA Analysis on all of our data to make new predictor variables that are independent. Each of these principal components explains a certain percentage of the variability within the data.

After doing the PCA:
We can then create a rotation matrix that shows how each of the PC variables weights our data used:

|               |       PC1 |       PC2 |       PC3 |       PC4 |
| :------------ | --------: | --------: | --------: | --------: |
| ht_scaled     |  0.241839 |  0.587699 |   -0.1721 |  -0.44664 |
| wt_scaled     |  0.262209 |   0.60891 |   -0.1084 |  0.076131 |
| forty_scaled  |  0.494082 |  0.025865 |  -0.31716 |  0.512618 |
| vertical_sc   |  -0.47238 |  0.288743 |  0.077587 |  0.308813 |
| bench_scaled  |  -0.14014 |  0.293846 |  0.212776 |   0.56319 |
| broad_jump    |  -0.40779 |  0.336803 |   0.26633 |   -0.1502 |
| cone_scaled   |  0.332729 | -0.002180 |  0.593606 |  0.216564 |
| shuttle_sc    |  0.329942 |  0.003259 |  0.619233 |  -0.22346 |

For example, PC1 weights the forty time, vertical, broad jump, three cone, and shuttle the heaviest. As PC1 increases, the fourty time increases, vertical decreases, broad jump distance decreases, and time it takes to complete the shuttle and cone increases, giving a slower, less explosive and athletic player. if PC1 decreases, you have a faster, more explosive and athletic player.

PC2 heavily weights the weight and height positively, so a large PC2 value represents a player with a much bigger frame, while small PC2 values repesents a player with a smaller frame.

PC3 weights the shuttle, cone, and fourty time drills the most. This separates players who mainly have straight line speed vs those who can accelerate in and out of breaks, and have good change of directions skills. A low PC3 value means someone with better lateral movement but slower, whereas a high PC3 value means someone faster with worse lateral movement.

PC4 weights the bench, forty time, and height. A high PC4 value corresponds to the shorter, slower, and stockier players whereas a low PC4 value corresponds to the faster, taller, and weaker player.

We also can show how much variance that each PC variable explains. Explaining 80% of the variance is a decent baseline, so we include PC1, PC2, PC3, PC4:

|     | explained_variance | explained_variance_ratio |
| :-- | -----------------: | -----------------------: |
| PC1 |           2.083091 |                 0.312541 |
| PC2 |           1.870914 |                 0.280706 |
| PC3 |           0.819527 |                 0.122959 |
| PC4 |           0.624784 |                 0.093741 |

Then using the PC variables, we can use k-means clustering to categorize the wide receivers into different clusters. For now, I chose to categorize into four different clusters. As we see each player gets matched with which cluster they belong to and the average values of the PC scores of the each cluster center, we can explain what type of wide receiver each cluster describes. Cluster 0 contains those who are athletic freaks. These are receivers who are heavy while still being fast and explosive, and usually slotted as your prototypical X. Examples of players would be DK Metcalf, Julio Jones, Xavier Legette. Those in cluster 1 are the speedy, shorter, lighter, explosive wide receivers like Xavier Worthy, Marquise Brown, Tyquan Thornton. Often, these are the players who are deep threats. Cluster 2 are wide receivers who are slower, but aren't very heavy and not as explosive. This contains players like Hunter Renfrow, Cooper Kupp, and Anquan Boldin. Cluster 3 contain the wide receivers who are bigger and taller, but are also less explosive and less athletic. Think DeAndre Hopkins or Elijah Sarratt from the current draft.

Then, we need to use UMAP to visualize these clusters into 2D space. Here is the result:

**Click on the image below to see how all the wide receivers are categorized into clusters based off all 4 PC variables, and then reduced to 2D using UMAP.**

[![Click to view UMAP plot.](resources/umap_ss.png)](https://etnguyen401.github.io/nfl-wr-clustering/data/wr_clusters_interactive_default.html)

As a bonus, we could also try to find clusters based off PC1 and PC2, but this would only explain about 60% of variance:

**Click on the image below to view the PC1 vs PC2 plot, which maps each wide receiver by PC1 (speed/explosiveness) and PC2(size/frame), and organizes them into clusters.**

[![Click to view UMAP plot.](resources/pc1_vs_pc2.png)](https://etnguyen401.github.io/nfl-wr-clustering/data/wr_clusters_pc1_vs_pc2.html)

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

**Click on the image below to see how all the wide receivers are categorized into clusters based off all 4 PC variables, and then reduced to 2D using UMAP.**

[![Click to view UMAP plot.](resources/umap_ss.png)](https://etnguyen401.github.io/nfl-wr-clustering/data/wr_clusters_interactive_default.html)

**Click on the image below to view the PC1 vs PC2 plot, which maps each wide receiver by PC1 (speed/explosiveness) and PC2(size/frame), and organizes them into clusters.**

[![Click to view UMAP plot.](resources/pc1_vs_pc2.png)](https://etnguyen401.github.io/nfl-wr-clustering/data/wr_clusters_pc1_vs_pc2.html)
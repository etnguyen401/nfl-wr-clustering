import os
from turtle import color
import pandas as pd
import nflreadpy as nfl
# from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans
import plotly.express as px
import umap

# combine_data = None
combine_data_types = {
    "wt": float, "forty": float, "vertical": float, "bench": float,
    "broad_jump": float, "cone": float, "shuttle": float,
    "ht-ft": float, "ht-in": float
}

if (os.path.exists("data/combine_data_wr_post_clean.csv")):
    print("Loading combine data from CSV...")
    combine_data = pd.read_csv("data/combine_data_wr_post_clean.csv", dtype=combine_data_types)
else:
    print("Scraping combine data from nflreadpy...")
    combine_data = nfl.load_combine().to_pandas()

    #filter for only wide receivers
    combine_data = combine_data.loc[(combine_data["pos"] == "WR")].reset_index(drop=True)
    #split height into feet and inches
    combine_data[["ht-ft", "ht-in"]] = combine_data["ht"].str.split("-", expand=True)

    # conver combine data to float
    combine_data = combine_data.astype(combine_data_types)

    # calculate height in terms of inches
    combine_data["ht"] = combine_data["ht-ft"] * 12.0 + combine_data["ht-in"]

    #remove unneeded cols
    combine_data.drop(["ht-ft", "ht-in"], axis=1, inplace=True)

    combine_data.to_csv("data/combine_data_wr_post_clean.csv", index=False)


print(combine_data.describe())

#fill in missing values in combine data
combine_data_imputed_path = "data/combine_data_wr_post_clean_imputed.csv"
cols_to_impute = ["ht", "wt", "forty", "vertical", "bench", "broad_jump", "cone", "shuttle"]
cols_to_impute_scaled = [col + "_scaled" for col in cols_to_impute]

if (os.path.exists(combine_data_imputed_path)):
    combine_data_imputed = pd.read_csv(combine_data_imputed_path)
else:
    # temp data frame with non-imputed cols
    temp_data = combine_data.drop(cols_to_impute, axis=1)

    # imputer = KNNImputer(n_neighbors=5, weights="distance")

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=1234),
        max_iter=10,
        random_state=1234
    )

    #scale data
    scaler = StandardScaler()
    combine_data_scaled = scaler.fit_transform(combine_data[cols_to_impute])

    # impute scaled data
    output_scaled = imputer.fit_transform(combine_data_scaled)
    output_scaled_df = pd.DataFrame(output_scaled, columns=cols_to_impute_scaled)

    #inverse scaled imputed data back to orignal scale
    output_og = scaler.inverse_transform(output_scaled)
    output_og_df = pd.DataFrame(output_og, columns=cols_to_impute)

    combine_data_imputed = pd.concat([temp_data, output_og_df, output_scaled_df], axis=1)
    combine_data_imputed.to_csv(combine_data_imputed_path, index=False)

print(combine_data_imputed.describe())

pca = PCA(n_components=4, svd_solver="full")
pca_fit = pca.fit_transform(combine_data_imputed[cols_to_impute_scaled])

#access PCs
pca_fit_data = pd.DataFrame(pca_fit)
pca_fit_data.columns = ["PC" + str(i + 1) for i in range(len(pca_fit_data.columns))]

combine_data_imputed = pd.concat([combine_data_imputed, pca_fit_data], axis=1)

rotation = pd.DataFrame(
    pca.components_.T,
    columns=["PC1", "PC2", "PC3", "PC4"],
    index=cols_to_impute_scaled,
)

print(f"Rotation matrix:\n{rotation}")
print(f"Explained variance: {pca.explained_variance_}")
pca_percent_py = pca.explained_variance_ratio_.round(4) * 100
print(f"Percent variance for each axis: {pca_percent_py}")

# sns.scatterplot(data=combine_data_imputed_scaled, x="PC1", y="PC2")
# plt.show()

# get clusters
k_means_fit_data = kmeans(combine_data_imputed[["PC1", "PC2", "PC3", "PC4"]], 4, rng=1234)

# add cluster col to data
combine_data_imputed["cluster"] = (
    vq(combine_data_imputed[["PC1", "PC2", "PC3", "PC4"]], k_means_fit_data[0])[0]
)
print("Cluster centers:")
print(k_means_fit_data[0])

combine_data_imputed["cluster"] = combine_data_imputed["cluster"].astype(str)

combine_data_imputed.to_csv("data/combine_data_wr_post_clean_imputed_pca_clusters.csv", index=False)

#for each cluster, get average value of each PC and print it out
print("Average values for each PC for each cluster:")
print(combine_data_imputed.groupby("cluster")[["PC1", "PC2", "PC3", "PC4"]].mean().round(2))

print("Average values for each feature for each cluster:")
print(combine_data_imputed.groupby("cluster")[cols_to_impute].mean().round(2))
cluster_values = sorted(combine_data_imputed["cluster"].unique())

# reduce dimensions of data using UMAP
reducer = umap.UMAP(random_state=1234)
umap_positions = reducer.fit_transform(combine_data_imputed[["PC1", "PC2", "PC3", "PC4"]])
centers_umap = reducer.transform(k_means_fit_data[0])

combine_data_imputed["UMAP1"] = umap_positions[:, 0]
combine_data_imputed["UMAP2"] = umap_positions[:, 1]

umap_fig = px.scatter(
    combine_data_imputed,
    x="UMAP1",
    y="UMAP2",
    title="Wide Receiver Clusters (UMAP)",
    color="cluster",
    category_orders={"cluster": cluster_values},
    hover_name="player_name",
    hover_data={
        "UMAP1": False,
        "UMAP2": False,
        "cluster": False,
        "ht": ":.2f",
        "wt": ":.2f",
        "forty": ":.2f",
        "vertical": ":.2f",
        "bench": ":.2f",
        "broad_jump": ":.2f",
        "cone": ":.2f",
        "shuttle": ":.2f",
    },
    labels={
        "UMAP1": "UMAP 1",
        "UMAP2": "UMAP 2",
        
    }
)

umap_fig.update_traces(
    marker=dict(size=8)
)

umap_fig.update_layout(
    hoverlabel=dict(bgcolor='white'),
)

#add cluster centers to umap_fig
umap_fig.add_scatter(
    x=centers_umap[:, 0],
    y=centers_umap[:, 1],
    mode="markers",
    marker=dict(size=16, color="black", symbol="star"),
    name="Cluster Centers",
    customdata=[0, 1, 2, 3],
    hovertemplate="Cluster %{customdata} Center <extra></extra>",
)

pc1_pc2_comp = px.scatter(
    combine_data_imputed,
    x="PC1",
    y="PC2",
    title="Speed/Explosiveness vs Size/Frame (PC1 vs PC2)",
    color="cluster",
    category_orders={"cluster": cluster_values},
    hover_name="player_name",
    hover_data={
        "PC1": False,
        "PC2": False,
        "cluster": False,
        "ht": ":.2f",
        "wt": ":.2f",
        "forty": ":.2f",
        "vertical": ":.2f",
        "bench": ":.2f",
        "broad_jump": ":.2f",
        "cone": ":.2f",
        "shuttle": ":.2f",
    },
    labels={
        "PC1": "PC1 - Measure of Speed and Explosiveness",
        "PC2": "PC2 - Measure of Size/Frame",
    }
)

pc1_pc2_comp.update_traces(
    marker=dict(size=8)
)

pc1_pc2_comp.update_layout(
    hoverlabel=dict(bgcolor='white'),
)

# add cluster centers to pc1_pc2_comp
pc1_pc2_comp.add_scatter(
    x=k_means_fit_data[0][:, 0],
    y=k_means_fit_data[0][:, 1],
    mode="markers",
    marker=dict(size=16, color="black", symbol="star"),
    name="Cluster Centers",
    customdata=[0, 1, 2, 3],
    hovertemplate="Cluster %{customdata} Center <extra></extra>",
)

umap_fig.write_html("data/wr_clusters_interactive_default.html")
pc1_pc2_comp.write_html("data/wr_clusters_pc1_vs_pc2.html")
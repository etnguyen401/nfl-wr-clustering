from pathlib import Path
import os
from turtle import color
import pandas as pd
import nflreadpy as nfl
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

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
COMBINE_DATA_PATH = DATA_DIR / "combine_data_wr_post_clean.csv"
IMPUTED_DATA_PATH = DATA_DIR / "combine_data_wr_post_clean_imputed.csv"
CLUSTER_CENTERS_PATH = DATA_DIR / "cluster_centers.csv"
ROTATION_PATH = DATA_DIR / "pca_rotation_matrix.csv"
PCA_VARIANCE_PATH = DATA_DIR / "pca_explained_variance.csv"
UMAP_HTML_PATH = DATA_DIR / "wr_clusters_interactive_default.html"
PC1_PC2_HTML_PATH = DATA_DIR / "wr_clusters_pc1_vs_pc2.html"
N_CLUSTERS = 4
RANDOM_STATE = 1234

combine_data_types = {
    "wt": float,
    "forty": float,
    "vertical": float,
    "bench": float,
    "broad_jump": float,
    "cone": float,
    "shuttle": float,
    "ht-ft": float,
    "ht-in": float,
}
cols_to_impute = ["ht", "wt", "forty", "vertical", "bench", "broad_jump", "cone", "shuttle"]
cols_to_impute_scaled = [col + "_scaled" for col in cols_to_impute]

def get_combine_data() -> pd.DataFrame:
    if COMBINE_DATA_PATH.exists():
        print("Loading combine data from CSV...")
        return pd.read_csv(COMBINE_DATA_PATH, dtype=combine_data_types)
    
    print("Scraping combine data from nflreadpy...")
    combine_data = nfl.load_combine().to_pandas()

    #filter for only wide receivers
    combine_data = combine_data.loc[combine_data["pos"] == "WR"].reset_index(drop=True)

    #split height into feet and inches
    combine_data[["ht-ft", "ht-in"]] = combine_data["ht"].str.split("-", expand=True)

    #convert combine data to float
    combine_data = combine_data.astype(combine_data_types)

    #calculate height in terms of inches
    combine_data["ht"] = combine_data["ht-ft"] * 12.0 + combine_data["ht-in"]

    #remove unneeded cols
    combine_data.drop(["ht-ft", "ht-in"], axis=1, inplace=True)

    combine_data.to_csv(COMBINE_DATA_PATH, index=False)
    return combine_data

def impute_combine_data(combine_data: pd.DataFrame) -> pd.DataFrame:
    if all(col in combine_data.columns for col in cols_to_impute_scaled):
        print("Imputed columns already exist in dataframe, skipping imputation...")
        return combine_data

    print("Imputing missing values in combine data...")
    temp_data = combine_data.drop(cols_to_impute, axis=1)
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE),
        max_iter=10,
        random_state=RANDOM_STATE,
    )

    #scale data
    scaler = StandardScaler()
    combine_data_scaled = scaler.fit_transform(combine_data[cols_to_impute])
    
    #impute scaled data
    output_scaled = imputer.fit_transform(combine_data_scaled)
    output_scaled_df = pd.DataFrame(output_scaled, columns=cols_to_impute_scaled)

    #inverse scaled imputed data back to orignal scale
    output_og = scaler.inverse_transform(output_scaled)
    output_og_df = pd.DataFrame(output_og, columns=cols_to_impute)

    combine_data_imputed = pd.concat([temp_data, output_og_df, output_scaled_df], axis=1)
    combine_data_imputed.to_csv(COMBINE_DATA_PATH, index=False)
    return combine_data_imputed

def add_pca_features(combine_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pca_cols = [f"PC{i + 1}" for i in range(N_CLUSTERS)]
    # if pca columns are in dataframe and other pca files exist, load them and return them
    if all(col in combine_data.columns for col in pca_cols) and ROTATION_PATH.exists() and PCA_VARIANCE_PATH.exists():
        #since cols are already in dataframe, we can just load the other two files and return them
        # combine_data_with_pca = pd.read_csv(COMBINE_DATA_PATH)
        print("PCA columns already exist in dataframe and rotation and explained variance files exist, loading from CSV...")
        rotation_df = pd.read_csv(ROTATION_PATH, index_col=0)
        explained_variance_df = pd.read_csv(PCA_VARIANCE_PATH, index_col=0)
        return combine_data, rotation_df, explained_variance_df

    print("Adding PCA features to combine data...")
    pca = PCA(n_components=4, svd_solver="full")
    pca_fit = pca.fit_transform(combine_data[cols_to_impute_scaled])
    pca_fit_df = pd.DataFrame(pca_fit, columns=pca_cols)
    combine_data_with_pca = pd.concat([combine_data, pca_fit_df], axis=1)
    
    # update combine data file
    combine_data_with_pca.to_csv(COMBINE_DATA_PATH, index=False)

    rotation_df = pd.DataFrame(
            pca.components_.T,
            columns=pca_cols,
            index=cols_to_impute_scaled,
    )
    # save rotation to a csv file
    rotation_df.to_csv(ROTATION_PATH, index=True)
    # save explained variance to csv
    explained_variance_df = pd.DataFrame(
        {
            "explained_variance": pca.explained_variance_,
            "explained_variance_ratio": pca.explained_variance_ratio_,
        },
        index=pca_cols,
    )
    explained_variance_df.to_csv(PCA_VARIANCE_PATH, index=True)
    return combine_data_with_pca, rotation_df, explained_variance_df

def get_clusters(combine_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # check if cluster column already exists in dataframe and if so, return the dataframe and the cluster centers
    if "cluster" in combine_data.columns and CLUSTER_CENTERS_PATH.exists():
        print("Cluster column already exists in dataframe and cluster centers file exists, loading from CSV...")
        cluster_centers_df = pd.read_csv(CLUSTER_CENTERS_PATH, index_col=0)

        return combine_data, cluster_centers_df

    print("Getting clusters for combine data...")
    # get clusters
    k_means_fit_data = kmeans(combine_data[["PC1", "PC2", "PC3", "PC4"]], N_CLUSTERS, rng=RANDOM_STATE)

    # add cluster col to data
    combine_data["cluster"] = vq(combine_data[["PC1", "PC2", "PC3", "PC4"]], k_means_fit_data[0])[0]

    combine_data["cluster"] = combine_data["cluster"].astype(str)

    combine_data.to_csv(COMBINE_DATA_PATH, index=False)

    # save cluster centers to a csv file
    cluster_centers_df = pd.DataFrame(
        k_means_fit_data[0],
        columns=["PC1", "PC2", "PC3", "PC4"],
        index=[f"Cluster {i}" for i in range(N_CLUSTERS)],
    )
    cluster_centers_df.to_csv(CLUSTER_CENTERS_PATH, index=True)

    return combine_data, cluster_centers_df

def build_figure(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    x_label: str,
    y_label: str,
    center_points: pd.DataFrame,
):
    # data = data.copy()
    data["cluster"] = data["cluster"].astype(str)
    cluster_values = sorted(data["cluster"].unique())
    hover_data = {
        x_column: False,
        y_column: False,
        "cluster": False,
        "ht": ":.2f",
        "wt": ":.2f",
        "forty": ":.2f",
        "vertical": ":.2f",
        "bench": ":.2f",
        "broad_jump": ":.2f",
        "cone": ":.2f",
        "shuttle": ":.2f",
    }

    figure = px.scatter(
        data,
        x=x_column,
        y=y_column,
        title=title,
        color="cluster",
        category_orders={"cluster": cluster_values},
        hover_name="player_name",
        hover_data=hover_data,
        labels={x_column: x_label, y_column: y_label},
        render_mode="svg",
    )

    figure.update_traces(
        marker=dict(size=8),
        legendgroup="clusters",
        legendgrouptitle_text="Clusters:",
    )

    #create centers
    center_labels = [f"Cluster {i} Center" for i in range(len(center_points))]
    center_figure = px.scatter(
        x=center_points[x_column],
        y=center_points[y_column],
        color=center_labels,
    )

    center_figure.update_traces(
        marker=dict(size=20, symbol="star", line=dict(width=2, color="black")),
        hovertemplate="%{fullData.name}<extra></extra>",
        legendgroup="centers",
        legendgrouptitle_text="Cluster Centers:",
    )

    figure.add_traces(list(center_figure.data))
    figure.data = figure.data[-len(center_labels):] + figure.data[:-len(center_labels)]
    figure.update_layout(
        hoverlabel=dict(bgcolor="white"),
        legend_tracegroupgap=24,
        legend_title_text="Legend",
    )
    return figure

def run_pipeline() -> None:
    combine_data = get_combine_data()
    print(combine_data.describe())

    # data after imputation
    combine_data_imputed = impute_combine_data(combine_data)
    print(combine_data_imputed.describe())

    combine_data_with_pca, rotation, explained_variance = add_pca_features(combine_data_imputed)
    print(f"Rotation matrix:\n{rotation}")
    print(f"Explained variance:\n{explained_variance}")

    combine_data_clustered, cluster_centers = get_clusters(combine_data_with_pca)
    print(f"Cluster centers:\n{cluster_centers}")

    print("Average values for each PC for each cluster:")
    print(combine_data_clustered.groupby("cluster")[["PC1", "PC2", "PC3", "PC4"]].mean().round(2))

    print("Average values for each feature for each cluster:")
    print(combine_data_clustered.groupby("cluster")[cols_to_impute].mean().round(2))

    reducer = umap.UMAP(random_state=RANDOM_STATE)
    umap_positions = reducer.fit_transform(combine_data_clustered[["PC1", "PC2", "PC3", "PC4"]])
    centers_umap = reducer.transform(cluster_centers)
    centers_umap_df = pd.DataFrame(centers_umap, columns=["UMAP1", "UMAP2"])

    combine_data_clustered["UMAP1"] = umap_positions[:, 0]
    combine_data_clustered["UMAP2"] = umap_positions[:, 1]

    umap_fig = build_figure(
        combine_data_clustered,
        x_column="UMAP1",
        y_column="UMAP2",
        title="Wide Receiver Clusters (UMAP)",
        x_label="UMAP 1",
        y_label="UMAP 2",
        center_points=centers_umap_df,
    )

    pc1_pc2_comp = build_figure(
        combine_data_clustered,
        x_column="PC1",
        y_column="PC2",
        title="Speed/Explosiveness vs Size/Frame (PC1 vs PC2)",
        x_label="PC1 - Measure of Speed and Explosiveness",
        y_label="PC2 - Measure of Size/Frame",
        center_points=cluster_centers[["PC1", "PC2"]],
    )

    umap_fig.write_html(UMAP_HTML_PATH)
    pc1_pc2_comp.write_html(PC1_PC2_HTML_PATH)


if __name__ == "__main__":
    run_pipeline()
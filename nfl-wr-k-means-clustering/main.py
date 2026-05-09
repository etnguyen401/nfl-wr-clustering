import os
import pandas as pd
import nflreadpy as nfl
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.vq import vq, kmeans

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

if (os.path.exists(combine_data_imputed_path)):
    combine_data_imputed = pd.read_csv(combine_data_imputed_path)
else:
    # temp data frame with non-imputed cols
    temp_data = combine_data.drop(cols_to_impute, axis=1)

    imputer = KNNImputer(n_neighbors=5, weights="distance")

    knn_output = imputer.fit_transform(combine_data[cols_to_impute])

    knn_output_df = pd.DataFrame(knn_output, columns=cols_to_impute)

    combine_data_imputed = pd.concat([temp_data, knn_output_df], axis=1)
    combine_data_imputed.to_csv(combine_data_imputed_path, index=False)

print(combine_data_imputed.describe())

# scale data and run PCA

combine_data_imputed_scaled = (
    combine_data_imputed[cols_to_impute] - \
    combine_data_imputed[cols_to_impute].mean()) / \
    combine_data_imputed[cols_to_impute].std() 

pca = PCA(svd_solver="full")
pca_fit = pca.fit_transform(combine_data_imputed_scaled)

rotation = pd.DataFrame(pca.components_, index=cols_to_impute)
print(f"Rotation matrix:\n{rotation}")
print(f"Explained variance: {pca.explained_variance_}")
pca_percent_py = pca.explained_variance_ratio_.round(4) * 100
print(f"Percent variance for each axis: {pca_percent_py}")

#access PCs
pca_fit_data = pd.DataFrame(pca_fit)
pca_fit_data.columns = ["PC" + str(i + 1) for i in range(len(pca_fit_data.columns))]

combine_data_imputed = pd.concat([combine_data_imputed, pca_fit_data], axis=1)

# sns.scatterplot(data=combine_data_imputed, x="PC1", y="PC2")
# plt.show()

# get clusters
k_means_fit_data = kmeans(combine_data_imputed[["PC1", "PC2"]], 4, seed=1234)

# add cluster col to data
combine_data_imputed["cluster"] = (
    vq(combine_data_imputed[["PC1", "PC2"]], k_means_fit_data[0])[0]
)

combine_data_imputed.to_csv("data/combine_data_wr_post_clean_imputed_pca_clusters.csv", index=False)

sns.scatterplot(
    data=combine_data_imputed,
    x="PC1",
    y="PC2",
    hue="cluster",
    palette="colorblind",
)
plt.show()
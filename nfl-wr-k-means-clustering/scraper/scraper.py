import os
import pandas as pd
import nflreadpy as nfl
from sklearn.impute import KNNImputer

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
    combine_data = combine_data.loc[(combine_data["pos"] == "WR")]
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

    imputer = KNNImputer(n_neighbors=10)

    knn_output = imputer.fit_transform(combine_data[cols_to_impute])

    knn_output_df = pd.DataFrame(knn_output, columns=cols_to_impute)

    combine_data_imputed = pd.concat([temp_data, knn_output_df], axis=1)
    combine_data_imputed.to_csv(combine_data_imputed_path, index=False)

print(combine_data_imputed.describe())
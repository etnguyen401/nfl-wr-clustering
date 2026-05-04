import os
import pandas as pd
import nflreadpy as nfl

combine_data = None

if (os.path.exists("data/combine_data_wr.csv")):
    print("Loading combine data from CSV...")
    combine_data = pd.read_csv("data/combine_data_wr.csv")
else:
    print("Scraping combine data from nflreadpy...")
    combine_data = nfl.load_combine().to_pandas()
    #filter for only wide receivers
    combine_data = combine_data.loc[(combine_data["pos"] == "WR")]
    combine_data.to_csv("data/combine_data_wr.csv", index=False)

print(combine_data.head())

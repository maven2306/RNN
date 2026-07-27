# %%


import pandas as pd

from pathlib import Path



"""
NB: the data consists of 18 input files and 18 output files. I will start manipulating only one input and one output file, and then scale up to all the data later. 
"""

train_data = pd.read_csv("/Users/matteo/GitHub/RNN/train/input_2023_w01.csv")
train_data[train_data["player_to_predict"] == True]

# Add check for whether there are any players still with False or "False" 

mask = train_data["play_direction"] == "left"
train_data.loc[mask, "x"] = 120 - train_data.loc[mask, "x"]
train_data.loc[mask, "ball_land_x"] = 120 - train_data.loc[mask, "ball_land_x"]
train_data.loc[mask, "dir"] = (train_data.loc[mask, "dir"] + 180) % 360
train_data.loc[mask, "o"] = (train_data.loc[mask, "o"] + 180) % 360

# %%



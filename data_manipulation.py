# %%


import pandas as pd

from pathlib import Path

import tensorflow as tf # only used for the Dataset construction 

import numpy as np

"""

The number of input frames (frame_id in the input files) and output frames (frame_id in the output files) is game_id dependent. 

The number of input players that need to be predicted (there are output frames available) is flagged "player_to_predict" = True 

"""


train = Path('/Users/matteo/GitHub/RNN/train')


input_dfs = []
output_dfs = []


for f in train.glob('input*'):
    input_dfs.append(pd.read_csv(f))
train_data = pd.concat(input_dfs, ignore_index=True)



for f in train.glob('output*'):
    output_dfs.append(pd.read_csv(f))
output_data = pd.concat(output_dfs, ignore_index=True)


    

# ============== Start of the code 

train_data = train_data[train_data["player_to_predict"] == True]

print(train_data["player_to_predict"].value_counts())
# Add check for whether there are any players still with False or "False" 

mask = train_data["play_direction"] == "left"
train_data.loc[mask, "x"] = 120 - train_data.loc[mask, "x"]
train_data.loc[mask, "ball_land_x"] = 120 - train_data.loc[mask, "ball_land_x"]
train_data.loc[mask, 'absolute_yardline_number'] = 100 - train_data.loc[mask, 'absolute_yardline_number']

# As 0 = east, 90 = north, 180 = west, 270 = south, we want to mirror degrees vertically (on a right-to-left play, a player facing north-west (160) would point north-east (20) on a left-to-right play)
train_data.loc[mask, "dir"] = (180 - train_data.loc[mask, "dir"]) % 360
train_data.loc[mask, "o"] = (180 - train_data.loc[mask, "o"]) % 360


# 360 = 0, but for NN is it not given. Circular variables are more complex to handle. Instead, we can use sin(theta) and cos(theta): sin(360) = sin(0), etc. 

for col in ["dir", "o"]:
    train_data[f"cos_{col}"] = np.cos(np.radians(train_data[col]))
    train_data[f"sin_{col}"] = np.sin(np.radians(train_data[col]))



# Add x transformation to the output file too! 

dir_map = train_data[["game_id", "play_id", "play_direction"]].drop_duplicates()
output_data = output_data.merge(dir_map, on=["game_id", "play_id"], how="left")

mask = output_data["play_direction"] == "left"
output_data.loc[mask, "x"] = 120 - output_data.loc[mask, "x"]

# Add tests 

def convert_height(h):
    feet, inches = h.split('-')
    return int(feet) * 30.48 + int(inches) * 2.54

train_data['player_height'] = train_data['player_height'].apply(convert_height)


print(train_data["player_height"].head(20))


def convert_birth_year(date):
    year, _, _ = date.split('-')
    return int(year)


train_data['player_birth_date'] = train_data['player_birth_date'].apply(convert_birth_year)
train_data = train_data.rename(columns={'player_birth_date':'player_birth_year'})


print(train_data["player_birth_year"])


category_cols = ['player_position', 'player_role']

train_data = pd.get_dummies(train_data, columns=category_cols, dtype=int)


# Now we want to create the masks for the data splitting. We must shuffle and split 70/15/15. I decide to split on game_id, so plays and players from training cannot leak to test/val 

unique_games = train_data["game_id"].unique()

np.random.seed(42)
np.random.shuffle(unique_games)

n_games = len(unique_games)
train_end = int(0.70 * n_games)
val_end   = int(0.85 * n_games)

train_games = unique_games[:train_end]
val_games   = unique_games[train_end:val_end]
test_games  = unique_games[val_end:]


# ---------- Now we're ready to create the tensors 


## We remove these columns from the input dataframes. In the output dataframes we only keep x and y so no columns have to be explicitly dropped 
cols_to_remove = ["game_id", "play_id", "nfl_id", "frame_id", 'play_direction', 'player_side',
                  'player_to_predict', 'num_frames_output', 'player_name', 'dir', 'o']

feature_cols = [c for c in train_data.columns if c not in cols_to_remove]
n_features = len(feature_cols)

def build_dataset(input_df, output_df):
    # This creates for every instance a dataframe. A groupby object is a dictionary where the key is one instance (nfl_id-play_id-game_id) and the value 
    # are the rows of that instance. 
    input_groups = input_df.groupby(["game_id", "play_id", "nfl_id"])
    output_groups = output_df.groupby(["game_id", "play_id", "nfl_id"])

    input_sequences = []
    output_sequences = []
    keys = []

    for key, grp in input_groups:
        grp = grp.sort_values("frame_id")
        input_sequences.append(grp[feature_cols].values.astype("float32"))
        keys.append(key)

    for key in keys:
        if key in output_groups.groups:
            grp = output_groups.get_group(key).sort_values("frame_id")
            output_sequences.append(grp[["x", "y"]].values.astype("float32")) # Here we keep only x and y from the output sequences 
        else:
            raise KeyError(f'{key} has no output data')
        
    # return one input sequence and one output sequence at time with yield 
    def gen():
        for inp, out in zip(input_sequences, output_sequences):
            yield inp, out


   
    dataset = tf.data.Dataset.from_generator(
        gen,                         # the generator function
        output_signature=(           # promise about what each yield looks like
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32), # n_features is the number of columns in the input sequences; None means: each tensor can have a variable length (different input frames number)
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32), # 2 is the number of columns in the output sequences; None means: each tensor can have a variable length (different output frames number)
        )
)
    # Now we batch the tensors and pad them: each batch has tensors of the same shape. Mind that different batches have different sizes: each batch has max_input_frame_id (because of None) 
    # as the input tensor, and max_output_frame_id (because of None) as the output tensor. 
    # We could also pad all sequences to a global max, but it would be inefficient. 

    dataset = dataset.padded_batch(
        batch_size=32,
        padded_shapes=([None, n_features], [None, 2]),
        padding_values=(999, 999)
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset 


train_input_df = train_data[train_data["game_id"].isin(train_games)]
train_output_df = output_data[output_data["game_id"].isin(train_games)]

val_input_df = train_data[train_data["game_id"].isin(val_games)]
val_output_df = output_data[output_data["game_id"].isin(val_games)]

test_input_df = train_data[train_data["game_id"].isin(test_games)]
test_output_df = output_data[output_data["game_id"].isin(test_games)]


# Let's first normalise and scale to 0 mean and 1 SD. We must use the same mean and std from the training split: any scaled value is dependent on the mean and std; 
# if birth_year(e.g. 1999) - mean = 0.34 and we train the model on that, if we scale the test data using a different mean, then 1999 - mean != 0.34! Same birth year, different coefficients. 
cols_to_normalize = ['absolute_yardline_number', 'player_height', 'player_weight', 'player_birth_year', 'x', 'y', 's', 'a',
                     'ball_land_x', 'ball_land_y']

mean = train_input_df.loc[:, cols_to_normalize].mean(axis=0)
std = train_input_df.loc[:, cols_to_normalize].std(axis=0)
train_input_df.loc[:, cols_to_normalize] = (train_input_df.loc[:, cols_to_normalize] - mean) / std
val_input_df.loc[:, cols_to_normalize] = (val_input_df.loc[:, cols_to_normalize] - mean) / std
test_input_df.loc[:, cols_to_normalize] = (test_input_df.loc[:, cols_to_normalize] - mean) / std


cols_to_normalize = ['x', 'y']

train_output_df.loc[:, cols_to_normalize] = (train_output_df.loc[:, cols_to_normalize] - mean) / std
val_output_df.loc[:, cols_to_normalize] = (val_output_df.loc[:, cols_to_normalize] - mean) / std
test_output_df.loc[:, cols_to_normalize] = (test_output_df.loc[:, cols_to_normalize] - mean) / std


"""
I think this is equivalent to: 

from sklearn.preprocessing import StandardScaler

# Fit scaler on training input only
input_scaler = StandardScaler()
input_scaler.fit(train_input_df[scale_cols])

# Transform all three splits with the SAME scaler
train_input_df[scale_cols] = input_scaler.transform(train_input_df[scale_cols])
val_input_df[scale_cols]   = input_scaler.transform(val_input_df[scale_cols])
test_input_df[scale_cols]  = input_scaler.transform(test_input_df[scale_cols])

etc

# NB: I will need output_scaler.inverse_transform(predictions) to transform back the predictions and get the RMSD in yards. 

"""


train_dataset = build_dataset(train_input_df, train_output_df)
val_dataset   = build_dataset(val_input_df, val_output_df)
test_dataset  = build_dataset(test_input_df, test_output_df)

print(len(train_games), len(val_games), len(test_games))

train_dataset.save('/Users/matteo/GitHub/RNN/datasets/train')
val_dataset.save('/Users/matteo/GitHub/RNN/datasets/val')
test_dataset.save('/Users/matteo/GitHub/RNN/datasets/test')


# %%

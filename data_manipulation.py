# %%


import pandas as pd

from pathlib import Path

import keras # Here only used for the splitting 

import tensorflow as tf # only used for the Dataset construction 



"""

The number of input frames (frame_id in the input files) and output frames (frame_id in the output files) is game_id dependent. 

The number of input players that need to be predicted (there are output frames available) is flagged "player_to_predict" = True 

"""


train = Path('/Users/matteo/GitHub/RNN/train')


train_data = pd.DataFrame()
output_data = pd.DataFrame()

for f in train.glob('input*'):
    df = pd.read_csv(f)
    train_data = pd.concat([train_data, df], ignore_index=False)


for f in train.glob('output*'):
    df = pd.read_csv(f)
    output_data = pd.concat([output_data, df], ignore_index=False)

    

# ============== Start of the code 

train_data = train_data[train_data["player_to_predict"] == True]

print(train_data["player_to_predict"].value_counts())
# Add check for whether there are any players still with False or "False" 

mask = train_data["play_direction"] == "left"
train_data.loc[mask, "x"] = 120 - train_data.loc[mask, "x"]
train_data.loc[mask, "ball_land_x"] = 120 - train_data.loc[mask, "ball_land_x"]
train_data.loc[mask, "dir"] = (train_data.loc[mask, "dir"] + 180) % 360
train_data.loc[mask, "o"] = (train_data.loc[mask, "o"] + 180) % 360

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
    return year


train_data['player_birth_date'] = train_data['player_birth_date'].apply(convert_birth_year)

print(train_data["player_birth_date"])


category_cols = ['player_position', 'player_role']

train_data = pd.get_dummies(train_data, columns=category_cols, dtype=int)







### Now we're ready to create the tensors 

# This creates for every instance a dataframe. A groupby object is a dictionary where the key is one instance (nfl_id-play_id-game_id) and the value 
# are the rows of that instance. 
input_groups = train_data.groupby(["game_id", "play_id", "nfl_id"])
output_groups = output_data.groupby(["game_id", "play_id", "nfl_id"])

for key, grp in input_groups:

    print(f'The key is: {key}')
    print(f'The group is: {grp}')
    print(f'Number of columns: {len(grp.columns)}')
    break


# We want to create lists of sequences that we will convert into tensors 

input_sequences = []
output_sequences = []
keys = []  # to keep track of (game_id, play_id, nfl_id)


# We remove these columns from the input dataframes. In the output dataframes we only keep x and y so no columns have to be explicitly dropped 
cols_to_remove = ["game_id", "play_id", "nfl_id", "frame_id", 'play_direction', 'player_side',
                  'player_to_predict', 'num_frames_output', 'player_name']

for key, grp in input_groups:
    # sort just in case
    grp = grp.sort_values("frame_id")
    # drop columns that are not features
    feature_cols = [c for c in grp.columns if c not in cols_to_remove]
    
    input_sequences.append(grp[feature_cols].values.astype("float32"))
    keys.append(key)

for key in keys:
    if key in output_groups.groups:
        grp = output_groups.get_group(key).sort_values("frame_id")
        output_sequences.append(grp[["x", "y"]].values.astype("float32"))
    else:
        raise KeyError(f'{key} has not output data')



# return one input sequence and one output sequence at time with yield 
def gen():
    for input_seq, output_seq in zip(input_sequences, output_sequences):
        yield input_seq, output_seq


print(input_sequences[0])

# N.B. A dataset object is NOT a tensor. It is a collection of tensors, similar to a list of tensors. So, tensors can take different shapes. 

dataset = tf.data.Dataset.from_generator(
    gen,                         # the generator function
    output_signature=(           # promise about what each yield looks like
        tf.TensorSpec(shape=(None, 27), dtype=tf.float32), # 27 is the number of columns in the input sequences; None means: each tensor can have a variable length (different input frames number)
        tf.TensorSpec(shape=(None, 2), dtype=tf.float32), # 2 is the number of columns in the output sequences; None means: each tensor can have a variable length (different output frames number)
    )
)


# Now we batch the tensors and pad them: each batch has tensors of the same shape. Mind that different batches have different sizes: each batch has max_input_frame_id (because of None) 
# as the input tensor, and max_output_frame_id (because of None) as the output tensor. 
# We could also pad all sequences to a global max, but it would be inefficient. 


# We also must shuffle and split 70/15/15. I decide to split on game_id, so plays and players from training cannot leak to test/val 

shuffled = dataset.shuffle(buffer_size=1000, seed=42)

train_ds, temp_ds = keras.utils.split_dataset(
    shuffled, left_size=0.70, right_size=0.30, seed=42
)
val_ds, test_ds = keras.utils.split_dataset(
    temp_ds, left_size=0.50, right_size=0.50, seed=42
)

# Now batch each split separately (padding per-split is more efficient)
train_ds = train_ds.padded_batch(32, padded_shapes=([None, 27], [None, 2])).prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.padded_batch(32,   padded_shapes=([None, 27], [None, 2])).prefetch(tf.data.AUTOTUNE)
test_ds  = test_ds.padded_batch(32,  padded_shapes=([None, 27], [None, 2])).prefetch(tf.data.AUTOTUNE)





# %%

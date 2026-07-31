# %%


import pandas as pd

from pathlib import Path

import tensorflow as tf # only used for the Dataset construction 

import numpy as np

"""

The number of input frames (frame_id in the input files) and output frames (frame_id in the output files) is game_id dependent. 

The number of input players that need to be predicted (there are output frames available) is flagged "player_to_predict" = True 

NB: the conda environment is RNN 
"""


train = Path('/Users/matteo/GitHub/RNN/train')


input_dfs = []
output_dfs = []


for f in train.glob('input*'):
    input_dfs.append(pd.read_csv(f))
train_data = pd.concat(input_dfs, ignore_index=True)


# Keep an unfiltered copy for verification
raw_train_data = train_data.copy()


for f in train.glob('output*'):
    output_dfs.append(pd.read_csv(f))
output_data = pd.concat(output_dfs, ignore_index=True)


train_data = train_data[train_data["player_to_predict"] == True]

# Snapshot for later assertions (after filtering, before transformations)
geo_snapshot = train_data[[
    "play_direction", "x", "y", "ball_land_x",
    "absolute_yardline_number", "dir", "o"
]].copy()

mask = train_data["play_direction"] == "left"
train_data.loc[mask, "x"] = 120 - train_data.loc[mask, "x"]
train_data.loc[mask, "ball_land_x"] = 120 - train_data.loc[mask, "ball_land_x"]
train_data.loc[mask, 'absolute_yardline_number'] = 100 - train_data.loc[mask, 'absolute_yardline_number']

# As 0 = north, 270 = east, 180 = south, 90 = west (all verified empirically), we want to mirror degrees vertically (on a right-to-left play, a player facing north-west (160) would point north-east (20) on a left-to-right play)
train_data.loc[mask, "dir"] = (360 - train_data.loc[mask, "dir"]) % 360
train_data.loc[mask, "o"] = (360 - train_data.loc[mask, "o"]) % 360


# 360 = 0, but for NN is it not given. Circular variables are more complex to handle. Instead, we can use sin(theta) and cos(theta): sin(360) = sin(0), etc. 

for col in ["dir", "o"]:
    train_data[f"cos_{col}"] = np.cos(np.radians(train_data[col]))
    train_data[f"sin_{col}"] = np.sin(np.radians(train_data[col]))



# Add x transformation to the output file too! 

dir_map = train_data[["game_id", "play_id", "play_direction"]].drop_duplicates()
output_data = output_data.merge(dir_map, on=["game_id", "play_id"], how="left")

# For assertions later
output_geo_snapshot = output_data[["play_direction", "x", "y"]].copy()

mask = output_data["play_direction"] == "left"
output_data.loc[mask, "x"] = 120 - output_data.loc[mask, "x"]

# Add tests 

def convert_height(h):
    feet, inches = h.split('-')
    return int(feet) * 30.48 + int(inches) * 2.54

# For assertions later
height_sample = train_data["player_height"].head(10).tolist()

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



train_input_df = train_data[train_data["game_id"].isin(train_games)]
train_output_df = output_data[output_data["game_id"].isin(train_games)]

val_input_df = train_data[train_data["game_id"].isin(val_games)]
val_output_df = output_data[output_data["game_id"].isin(val_games)]

test_input_df = train_data[train_data["game_id"].isin(test_games)]
test_output_df = output_data[output_data["game_id"].isin(test_games)]


# Let's first normalise and scale to 0 mean and 1 SD. We must use the same mean and std from the training split: any scaled value is dependent on the mean and std; 
# if birth_year(e.g. 1999) - mean = 0.34 and we train the model on that, if we scale the test data using a different mean, then 1999 - mean != 0.34! Same birth year, different coefficients. 


from sklearn.preprocessing import StandardScaler

cols_to_scale = ['absolute_yardline_number', 'player_height', 'player_weight', 'player_birth_year', 's', 'a',
                     'ball_land_x', 'ball_land_y']

scaler = StandardScaler()

scaler.fit(train_input_df[cols_to_scale]) # without x and y 

train_input_df[cols_to_scale] = scaler.transform(train_input_df[cols_to_scale])
val_input_df[cols_to_scale]   = scaler.transform(val_input_df[cols_to_scale])
test_input_df[cols_to_scale]  = scaler.transform(test_input_df[cols_to_scale])

cols_to_scale = ['x', 'y']

scaler_coords = StandardScaler()

scaler_coords.fit(train_input_df[cols_to_scale]) # A scaler just for x and y, as it must be applied, the same scaler, to both input and output datasets

train_input_df[cols_to_scale] = scaler_coords.transform(train_input_df[cols_to_scale])
val_input_df[cols_to_scale]   = scaler_coords.transform(val_input_df[cols_to_scale])
test_input_df[cols_to_scale]  = scaler_coords.transform(test_input_df[cols_to_scale])

train_output_df[cols_to_scale] = scaler_coords.transform(train_output_df[cols_to_scale])
val_output_df[cols_to_scale] = scaler_coords.transform(val_output_df[cols_to_scale])
test_output_df[cols_to_scale] = scaler_coords.transform(test_output_df[cols_to_scale])

# Persist the coordinate scaler so the notebook can convert predictions back to yards
import joblib
joblib.dump(scaler_coords, '/Users/matteo/GitHub/RNN/datasets/scaler_coords.joblib')

# NB: I will need output_scaler.inverse_transform(predictions) to transform back the predictions and get the RMSD in yards. 


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

    def shift_target(x, y):
        start = tf.zeros((1, 2), dtype=tf.float32)
        dec_in = tf.concat([start, y[:-1]], axis=0)
        return {"enc_in": x, "dec_in": dec_in}, y   

    
    dataset = tf.data.Dataset.from_generator(
        gen,                         # the generator function
        output_signature=(           # promise about what each yield looks like
            tf.TensorSpec(shape=(None, n_features), dtype=tf.float32), # n_features is the number of columns in the input sequences; None means: each tensor can have a variable length (different input frames number)
            tf.TensorSpec(shape=(None, 2), dtype=tf.float32), # 2 is the number of columns in the output sequences; None means: each tensor can have a variable length (different output frames number)
        )
)    

    dataset = dataset.map(shift_target)
    
    # Now we batch the tensors and pad them: each batch has tensors of the same shape. Mind that different batches have different sizes: each batch has max_input_frame_id (because of None) 
    # as the input tensor, and max_output_frame_id (because of None) as the output tensor. 
    # We could also pad all sequences to a global max, but it would be inefficient. 

    dataset = dataset.padded_batch(
        batch_size=512,
        padded_shapes=({"enc_in": [None, n_features], "dec_in": [None, 2]}, [None, 2]),
        padding_values=({"enc_in": 999.0, "dec_in": 999.0}, 999.0)
    )

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset 

train_dataset = build_dataset(train_input_df, train_output_df)
val_dataset   = build_dataset(val_input_df, val_output_df)
test_dataset  = build_dataset(test_input_df, test_output_df)

print(len(train_games), len(val_games), len(test_games))

train_dataset.save('/Users/matteo/GitHub/RNN/datasets/train')
val_dataset.save('/Users/matteo/GitHub/RNN/datasets/val')
test_dataset.save('/Users/matteo/GitHub/RNN/datasets/test')



# =========================================================
# BOTTOM-OF-SCRIPT ASSERTIONS
# =========================================================
print("\n--- Running bottom-of-script assertions ---\n")
eps = 1e-5


# 1. Loading & concatenation
input_files = list(train.glob('input*'))
output_files = list(train.glob('output*'))
assert len(input_files) > 0, "No input files found"
assert len(output_files) > 0, "No output files found"
assert len(input_files) == len(input_dfs), "Did not read all input files"
assert len(output_files) == len(output_dfs), "Did not read all output files"
assert len(raw_train_data) == sum(len(df) for df in input_dfs), "Input rows lost"
assert len(train_data) == sum(len(df[df["player_to_predict"] == True]) for df in input_dfs), "Filtered input rows mismatch"
assert len(output_data) == sum(len(df) for df in output_dfs), "Output rows lost"


# 2. Filtering: only players with output data are kept
assert train_data["player_to_predict"].all(), "Kept non-predict players"

input_keys = train_data[["game_id", "play_id", "nfl_id"]].drop_duplicates()
output_keys = output_data[["game_id", "play_id", "nfl_id"]].drop_duplicates()
merged = input_keys.merge(
    output_keys, on=["game_id", "play_id", "nfl_id"], how="outer", indicator=True
)
assert (merged["_merge"] == "both").all(), "Input/output keys do not match"


# 3. Geometric flip for left-direction plays
left = geo_snapshot[geo_snapshot["play_direction"] == "left"]
right = geo_snapshot[geo_snapshot["play_direction"] == "right"]

assert np.allclose(train_data.loc[left.index, "x"], 120 - left["x"], atol=eps), "x flip wrong"
assert np.allclose(train_data.loc[left.index, "ball_land_x"], 120 - left["ball_land_x"], atol=eps), "ball_land_x flip wrong"
assert np.allclose(train_data.loc[left.index, "absolute_yardline_number"], 100 - left["absolute_yardline_number"], atol=eps), "yardline flip wrong"
assert np.allclose(train_data.loc[left.index, "y"], left["y"], atol=eps), "y should not be flipped"
assert np.allclose(train_data.loc[left.index, "dir"], (360 - left["dir"]) % 360, atol=eps, equal_nan=True), "dir mirror wrong"
assert np.allclose(train_data.loc[left.index, "o"], (360 - left["o"]) % 360, atol=eps, equal_nan=True), "o mirror wrong"

# Right-direction plays should be untouched
assert np.allclose(train_data.loc[right.index, "x"], right["x"], atol=eps, equal_nan=True), "x changed for right plays"
assert np.allclose(train_data.loc[right.index, "ball_land_x"], right["ball_land_x"], atol=eps, equal_nan=True), "ball_land_x changed for right plays"
assert np.allclose(train_data.loc[right.index, "absolute_yardline_number"], right["absolute_yardline_number"], atol=eps, equal_nan=True), "yardline changed for right plays"
assert np.allclose(train_data.loc[right.index, "dir"], right["dir"], atol=eps, equal_nan=True), "dir changed for right plays"
assert np.allclose(train_data.loc[right.index, "o"], right["o"], atol=eps, equal_nan=True), "o changed for right plays"


# 4. sin/cos encoding
for col in ["dir", "o"]:
    assert f"cos_{col}" in train_data.columns, f"cos_{col} missing"
    assert f"sin_{col}" in train_data.columns, f"sin_{col} missing"
    assert np.allclose(train_data[f"cos_{col}"], np.cos(np.radians(train_data[col])), atol=eps, equal_nan=True)
    assert np.allclose(train_data[f"sin_{col}"], np.sin(np.radians(train_data[col])), atol=eps, equal_nan=True)


# 5. Output x flip
assert output_data["play_direction"].notna().all(), "Some output rows have no play_direction"

left_out = output_geo_snapshot[output_geo_snapshot["play_direction"] == "left"]
right_out = output_geo_snapshot[output_geo_snapshot["play_direction"] == "right"]

assert np.allclose(output_data.loc[left_out.index, "x"], 120 - left_out["x"], atol=eps), "Output x flip wrong"
assert np.allclose(output_data.loc[right_out.index, "x"], right_out["x"], atol=eps), "Output x changed for right plays"
assert np.allclose(output_data.loc[left_out.index, "y"], left_out["y"], atol=eps), "Output y should not be flipped"
assert np.allclose(output_data.loc[right_out.index, "y"], right_out["y"], atol=eps), "Output y changed for right plays"


# 6. Height conversion to cm
expected_cm = [int(h.split('-')[0]) * 30.48 + int(h.split('-')[1]) * 2.54 for h in height_sample]
assert np.allclose(train_data["player_height"].head(10).to_numpy(), expected_cm, atol=1e-6), "Height conversion to cm wrong"


# 7. Birth year
assert train_data["player_birth_year"].dtype in (np.int32, np.int64), "Birth year not integer"
assert train_data["player_birth_year"].between(1900, 2020).all(), "Birth year out of range"


# 8. One-hot encoding
assert "player_position" not in train_data.columns, "player_position not one-hot encoded"
assert "player_role" not in train_data.columns, "player_role not one-hot encoded"
assert any(c.startswith("player_position_") for c in train_data.columns), "No player_position dummies"
assert any(c.startswith("player_role_") for c in train_data.columns), "No player_role dummies"
assert train_data.filter(regex=r"^player_position_").isin([0, 1]).all().all(), "Position dummies not 0/1"
assert train_data.filter(regex=r"^player_role_").isin([0, 1]).all().all(), "Role dummies not 0/1"


# 9. Splits: no game leakage
assert len(set(train_games) & set(val_games)) == 0, "Train/val overlap"
assert len(set(train_games) & set(test_games)) == 0, "Train/test overlap"
assert len(set(val_games) & set(test_games)) == 0, "Val/test overlap"
assert len(set(train_games) | set(val_games) | set(test_games)) == n_games, "Games missing"

assert abs(len(train_games) / n_games - 0.70) < 0.02, "Train split not ~70%"
assert abs(len(val_games) / n_games - 0.15) < 0.02, "Val split not ~15%"
assert abs(len(test_games) / n_games - 0.15) < 0.02, "Test split not ~15%"

for name, inp, out, games in [
    ("train", train_input_df, train_output_df, train_games),
    ("val", val_input_df, val_output_df, val_games),
    ("test", test_input_df, test_output_df, test_games),
]:
    assert inp["game_id"].isin(games).all(), f"{name} input contains other games"
    assert out["game_id"].isin(games).all(), f"{name} output contains other games"


# 10. Scaling
for df in [train_input_df, val_input_df, test_input_df,
           train_output_df, val_output_df, test_output_df]:
    assert df[["x", "y"]].isna().sum().sum() == 0, "NaNs in coordinates after scaling"

first_scaled_cols = [c for c in scaler.feature_names_in_ if c not in ("x", "y")]
assert np.allclose(train_input_df[first_scaled_cols].mean(), 0, atol=1e-6), "First scaler means not zero"
assert np.allclose(train_input_df[first_scaled_cols].std(ddof=0), 1, atol=1e-6), "First scaler std not one"
assert np.allclose(train_input_df[["x", "y"]].mean(), 0, atol=1e-6), "Coord scaler mean not zero"
assert np.allclose(train_input_df[["x", "y"]].std(ddof=0), 1, atol=1e-6), "Coord scaler std not one"


# 11. Sequence alignment: same keys, each sorted by frame_id
input_groups = train_data.groupby(["game_id", "play_id", "nfl_id"])
output_groups = output_data.groupby(["game_id", "play_id", "nfl_id"])
assert set(input_groups.groups.keys()) == set(output_groups.groups.keys()), "Final input/output keys mismatch"

for key in input_groups.groups.keys():
    in_frames = input_groups.get_group(key).sort_values("frame_id")["frame_id"].tolist()
    out_frames = output_groups.get_group(key).sort_values("frame_id")["frame_id"].tolist()
    assert in_frames == sorted(in_frames), f"Input frames not sorted for {key}"
    assert out_frames == sorted(out_frames), f"Output frames not sorted for {key}"


# 12. Saved datasets load and have expected shape/padding
for split in ["train", "val", "test"]:
    path = Path(f'/Users/matteo/GitHub/RNN/datasets/{split}')
    assert path.exists(), f"{split} dataset not saved"

train_loaded = tf.data.Dataset.load('/Users/matteo/GitHub/RNN/datasets/train')
in_lengths = train_input_df.groupby(["game_id", "play_id", "nfl_id"]).size().tolist()
out_lengths = train_output_df.groupby(["game_id", "play_id", "nfl_id"]).size().tolist()

for b_idx, (batch, batch_out) in enumerate(train_loaded.take(1)):
    batch_enc = batch["enc_in"]
    batch_dec = batch["dec_in"]
    assert batch_enc.shape[2] == n_features, "Input feature dimension wrong"
    assert batch_dec.shape[2] == 2, "Decoder input dimension wrong"
    assert batch_out.shape[2] == 2, "Output dimension wrong"
    for i in range(batch_enc.shape[0]):
        idx = b_idx * 32 + i
        if idx >= len(in_lengths):
            break
        if in_lengths[idx] < batch_enc.shape[1]:
            assert (batch_enc[i, in_lengths[idx]:, :] == 999.0).numpy().all(), "Input padding not 999.0"
        if out_lengths[idx] < batch_dec.shape[1]:
            assert (batch_dec[i, out_lengths[idx]:, :] == 999.0).numpy().all(), "Decoder input padding not 999.0"
        if out_lengths[idx] < batch_out.shape[1]:
            assert (batch_out[i, out_lengths[idx]:, :] == 999.0).numpy().all(), "Output padding not 999.0"

print("\nAll bottom-of-script assertions passed.\n")

# %%

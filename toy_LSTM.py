# TOY — LSTM encoder-decoder with teacher forcing, batch-specific padding
import numpy as np, tensorflow as tf, keras
from keras import layers

np.random.seed(0)
N = 200; F_IN = 3; F_OUT = 2; PAD = 999.0
F_ENC = 16; F_DEC = 16

def make_sequences(n):
    X, Y, Lout = [], [], []
    for _ in range(n):
        ti = np.random.randint(4, 8)      # input length varies 4..7
        to = np.random.randint(3, 9)      # output length varies 3..8 (NO global cap)
        last = np.random.randn(F_IN)
        seq_in = np.random.randn(ti, F_IN).astype("float32"); seq_in[-1] = last
        seq_out = np.stack([last[0] + 0.10*np.arange(to),
                            last[1] - 0.05*np.arange(to)], axis=1).astype("float32")
        X.append(seq_in); Y.append(seq_out); Lout.append(to)
    return X, Y, np.array(Lout)

Xtr, Ytr, Ltr = make_sequences(N)
Xva, Yva, Lva = make_sequences(40)

# --- Build the shifted decoder input (teacher forcing) in the data pipeline.
#     dec_in[t] = y[t-1], dec_in[0] = START (zeros). Same length as y.
def shift_target(x, y):
    start = tf.zeros((1, F_OUT), dtype=tf.float32)
    dec_in = tf.concat([start, y[:-1]], axis=0)
    return {"enc_in": x, "dec_in": dec_in}, y         

def make_ds(X, Y, batch=32, shuffle=True):
    def gen():
        for x, y in zip(X, Y):
            yield x.astype("float32"), y.astype("float32")
    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(tf.TensorSpec((None, F_IN), tf.float32),
                          tf.TensorSpec((None, F_OUT), tf.float32)))
    ds = ds.map(shift_target)     # now yields (x, dec_in, y)
    if shuffle: ds = ds.shuffle(100)
    ds = ds.padded_batch(
        batch,
        padded_shapes=({"enc_in": [None, F_IN], "dec_in": [None, F_OUT]}, [None, F_OUT]),
        padding_values=({"enc_in": PAD, "dec_in": PAD}, PAD)
    )
    return ds.prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(Xtr, Ytr); val_ds = make_ds(Xva, Yva, shuffle=False)

def masked_mse(y_true, y_pred):
    mask = tf.cast(tf.reduce_all(y_true != PAD, axis=-1), tf.float32)   # (batch, T)
    sq = tf.reduce_sum(tf.square(y_true - y_pred), axis=-1)             # (batch, T)
    return tf.reduce_sum(sq * mask) / (tf.reduce_sum(mask) * F_OUT)

# --- Shared layer INSTANCES (so training model and inference model share weights)
enc_lstm = layers.LSTM(F_ENC, return_state=True, name="encoder_lstm")
dec_lstm = layers.LSTM(F_DEC, return_sequences=True, return_state=True, name="decoder_lstm")
dec_dense = layers.Dense(F_OUT, name="decoder_dense")

# --- TRAINING MODEL: encoder → states; decoder consumes shifted targets + those states
enc_in = keras.Input(shape=(None, F_IN), name="enc_in")
_, h, c = enc_lstm(layers.Masking(mask_value=PAD)(enc_in))   # (h, c) = encoder end state

dec_in = keras.Input(shape=(None, F_OUT), name="dec_in")
dec_out, _, _ = dec_lstm(layers.Masking(mask_value=PAD)(dec_in), initial_state=[h, c])
dec_out = dec_dense(dec_out)

model = keras.Model([enc_in, dec_in], dec_out)
model.compile(optimizer=keras.optimizers.Adam(1e-2), loss=masked_mse)
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=50, verbose=2)

# --- INFERENCE HELPERS: split the trained model into two reusable pieces.
#     Same layer instances => SAME weights as the training model. This is the crucial reuse.
encoder_model = keras.Model(enc_in, [h, c])                  # (T_in, F_IN) -> [h, c]

dec_in_step = keras.Input(shape=(1, F_OUT))                  # one timestep at a time
dec_h_in    = keras.Input(shape=(F_DEC,))
dec_c_in    = keras.Input(shape=(F_DEC,))
dec_out_step, dec_h_out, dec_c_out = dec_lstm(dec_in_step,   # no Masking needed: 1 real step
                                               initial_state=[dec_h_in, dec_c_in])
dec_pred_step = dec_dense(dec_out_step)
decoder_step = keras.Model([dec_in_step, dec_h_in, dec_c_in],
                           [dec_pred_step, dec_h_out, dec_c_out])

def predict_autoregressive(x, true_len):
    # 1) encode the whole input once → starting (h, c)
    h, c = encoder_model(x[None])                       # x[None] adds batch dim
    # 2) roll out one step at a time, feeding each prediction back in
    step = np.zeros((1, 1, F_OUT), dtype="float32")     # START token
    h = np.asarray(h); c = np.asarray(c)
    preds = []
    for _ in range(true_len):
        p, h, c = decoder_step([step, h, c])
        p = np.asarray(p); h = np.asarray(h); c = np.asarray(c)
        preds.append(p[0, 0])
        step = p                                        # AUTOREGRESSIVE: own output becomes next input
    return np.stack(preds)                              # (true_len, 2)

rmses = [np.sqrt(np.mean((predict_autoregressive(Xva[i], Lva[i]) - Yva[i])**2))
         for i in range(len(Xva))]
print("toy autoregressive RMSE:", np.mean(rmses))
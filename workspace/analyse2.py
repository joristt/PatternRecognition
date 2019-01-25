import pandas as pd
import numpy  as np
import os
import matplotlib.pyplot as plt
import sqlite3
from   time                  import time
from   sklearn.preprocessing import MinMaxScaler
from   sklearn.model_selection import train_test_split

# In debug mode, only a small portion of the 629.145.481 rows from train.csv
# will be read.
DEBUG = False

# DATA
# https://www.kaggle.com/c/LANL-Earthquake-Prediction/data
# 
# train.csv contains 629.145.481 rows and is ~10GB
# 
data_dir = "./data/"
train_path = data_dir + "train.csv"

# conn = sqlite3.connect(data_dir + "train.db")
# # conn.text_factory = str
# cur = conn.cursor()

# # A REAL in SQLite is an 8byte float, so we don't lose precision.
# def get_rows(l_bound):
#     cur.execute("SELECT * FROM train_table limit %s, %s" % (l_bound, l_bound + num - 1))
#     return cur.fetchall()








# mins = [np.float64(1000000)  for _ in range(2)]
# maxs = [np.float64(-1000000) for _ in range(2)]
# with open(train_path) as f:
#     lineNr = 0
#     f.readline() # Skip header row
#     for line in f:
#         if lineNr % 1000000 == 0:
#             print(lineNr)
#         lineNr += 1
#         elems = line.split(",")
#         for i in range(2):
#             mins[i] = min(mins[i], np.float64(elems[i]))
#             maxs[i] = max(maxs[i], np.float64(elems[i]))



s = time()
print("Loading data into RAM...")
# d = pd.read_csv(
# 	train_path,
# 	nrows=10e5 if DEBUG else None, # nrows=None causes all lines to be read.
# 	dtype = {
#         # The seismic signal is int16, but we normalise it later on and
#         # therefore we load the acoustic data as floats.
# 		"acoustic_data"   : np.float64,
# 		"time_to_failure" : np.float64 # The time (s) till next earthquake.
# 	}
# )
d = np.load("./data/train.npy")
print("Loaded data in %s seconds." % (time() - s))

# s = time()
# print("Normalising data...")
# Convert pd.dataframe to normalised np.ndarray in range [0,1]
# scaler = MinMaxScaler(feature_range=(0, 1))
# d = scaler.fit_transform(d)
# print("Normalised data in %s seconds." % (time() - s))

# Split into train and test data.
# split_i    = int((d.shape[0]) * 0.8)
# train_data = d[:split_i]
# d = d.values

# scalers = [MinMaxScaler(feature_range=(0, 1)) for i in range(2)]
# scalers[0].fit_transform(d.values[:,0])
# scalers[1].fit_transform(d.values[:,1])

# Calculate summary statistics per time step.
def summary_statistics(z):
	return np.c_[z.mean(axis=1), 
	             np.median(np.abs(z), axis=1),
	             z.std(axis=1), 
	             z.max(axis=1),
	             z.min(axis=1)]

def stats(z):
    return np.c_[
        z.mean(axis=1),
        z.max(axis=1),
        z.min(axis=1),
        z.var(axis=1)
    ]

# Calculates list of eartquake moments. The list contains tuples. Every tuple
# contains two indices. The first index is the last moment before the earth
# quake, the second index is always simply first index + 1, and is therefore
# the first moment after the earth quake.
def quake_moments(data):
	ms = np.where(data[1:,1] > data[:-1,1])[0]
	return zip(ms, ms+1)

# def plott(data, l_bound, u_bound, step_size):
# 	smpl       = data[l_bound:u_bound:step_size]

# 	# print(np.where(data[,0] > 800))

# 	quake_moms = quake_moments(data[l_bound:u_bound])
# 	# x-axis of plot is time offset, which is the sum of all time_to_failure
# 	# values of the first moments after the earth quakes.
# 	dur = data[0,1] + sum([data[qm,1] for _,qm in quake_moms])
# 	y_vals = np.linspace(0, dur, len(smpl))
# 	ts = pd.Series(smpl[l_bound:u_bound,0], index=y_vals)
# 	ts.plot()
# 	plt.show()

# plott(data, 0, 20000000, 10000)

def sample(data, size=100000):
    assert size <= len(data)
    return np.random.choice(
           data,
           size,
           replace=False
    )

resolution = 4

# For a given ending position "u_bound", we split the last 150.000 values of "x" into 150 pieces of length 1000 each.
# From each piece, 16 features are extracted. This results in a feature matrix of dimension (150 time steps x 16 features). 
def features(data, u_bound):
    l_bound = u_bound - 150000
    assert l_bound >= 0
    arr = np.array(data[l_bound:u_bound][::resolution])
    return np.expand_dims(arr, 1)
    # Reshaping and standardize approximately
    # l = data[l_bound:u_bound].reshape(nr_subsets, -1)
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    # return stats(l)
    # return np.c_[
    # 	summary_statistics(l),
    # 	summary_statistics(l[:, -subset_size // 10:]),
    # 	summary_statistics(l[:, -subset_size // 100:]),
    # 	l[:, -1:]
    # ]

# We call "summary_statistics" three times, so the total number of features is 3 * 5 + 1 (last value) = 16
# nr_features = 4

# Generates 32 indices in the range [mi,ma) that are all at least
# 150.000 values apart, such that the sequences they represent will never
# overlap. This function is technically not guaranteed to terminate, but
# will as long as parameters are decent.
def indices(mi, ma, nr, min_dist):
    while True:
        vs = np.sort(np.random.randint(mi, ma, nr))
        dists = np.all(vs[1:] - vs[:-1] >= min_dist)
        return vs
    

fst = features(
    d[:, 0],
    u_bound     = 74803393
)


# The generator randomly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "features".
def generator(data):
    l_bound     = 0
    u_bound     = len(data) - 1
    n_steps     = 1500
    step_length = 100

    batch_size  = 32
    while True:
        # Pick indices of ending positions
        # rows = np.random.randint(l_bound + n_steps * step_length, u_bound, size=batch_size)
        # rows = indices(0, len(d), batch_size, 150000)
        rows = np.random.randint(0, len(data), batch_size)
        # Initialize feature matrices and targets
        samples = np.empty((batch_size, 150000//resolution, 1)) # np.zeros((batch_size, n_steps, nr_features))
        targets = np.empty(batch_size, )
        for j, row in enumerate(rows):
            fts = features(
                data[:, 0],
                u_bound     = row
            )
            samples[j] = fts
            targets[j] = data[row, 1]
        yield samples, targets

train_gen = generator(d[:-len(d)//5])
valid_gen = generator(d[-len(d)//5:])












def resnet_layer(model,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    if conv_first:
        model.add(conv)
        if batch_normalization:
            model.add(BatchNormalization)
        if activation is not None:
            model.add(Activation(activation))
    else:
        if batch_normalization:
            model(BatchNormalization)
        if activation is not None:
            model.add(Activation(activation))
        model.add(conv(x))











# Define model
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential
from keras.layers import Dense, GRU, LSTM, Conv1D, BatchNormalization, GlobalAveragePooling1D
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

# keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
# keras.layers.      GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)

cb = [ModelCheckpoint("model.hdf5", monitor='val_loss', save_weights_only=False, period=3)]

model = Sequential()

model.add(Conv1D(128, kernel_size=8, strides=1, input_shape=(37500, 1)))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv1D(128, kernel_size=5, strides=1))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv1D(128, kernel_size=3, strides=1))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(GlobalAveragePooling1D())

model.add(Dense(1))

model.summary()

# Compile and fit model
model.compile(optimizer=adam(lr=0.0005), loss="mae")

history = model.fit_generator(train_gen,
                              steps_per_epoch=250,#n_train // batch_size,
                              epochs=1,
                              verbose=1,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=25)#n_val









# model = Sequential()
# model.add(resnet_layer(model, 64))
# model.add(resnet_layer(model, 64))
# model.add(resnet_layer(model, 64))

# model.add(resnet_layer(model, 128))
# model.add(resnet_layer(model, 128))
# model.add(resnet_layer(model, 128))

# model.add(resnet_layer(model, 128))
# model.add(resnet_layer(model, 128))
# model.add(resnet_layer(model, 128))











# model = Sequential()
# # NVIDIA Only: model.add(CuDNNGRU(48, input_shape=(None, nr_features)))
# model.add(GRU(48, input_shape=(37500,1))) # (150 (None mag ook), nr_features)))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(1))

# model.summary()

# # Compile and fit model
# model.compile(optimizer=adam(lr=0.0005), loss="mae")

# history = model.fit_generator(train_gen,
#                               steps_per_epoch=250,#n_train // batch_size,
#                               epochs=1,
#                               verbose=1,
#                               callbacks=cb,
#                               validation_data=valid_gen,
#                               validation_steps=25)#n_valid // batch_size)

# submission = pd.read_csv('data/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# # Load each test data, create the feature matrix, get numeric prediction
# for i, seg_id in enumerate(submission.index):
#     if i < 2607:
#         continue
#     print(i)
#     seg = pd.read_csv('data/test/' + seg_id + '.csv')
#     x = seg['acoustic_data'].values
#     submission.time_to_failure[i] = model.predict(np.expand_dims(features(x), 0))

# def inverse(submission):
#     # Invert scaling of submission.time_to_failure
#     ttfs = np.expand_dims(submission.time_to_failure.values, 1)
#     ttfs = np.append(ttfs, ttfs, axis=1)
#     ttfs = scaler.inverse_transform(ttfs)
#     ttfs = np.delete(ttfs, 0, 1)
#     submission.time_to_failure = np.clip(ttfs, 0, a_max=None)
#     submission.to_csv("data/submission.csv")


# # # predictions = model.predict_on_batch(next(gen)[0])
# # predictions=. submission.time_to_failure.values
# # ttfs = predictions
# # ttfs = np.append(ttfs, ttfs, axis=1)
# # ttfs = scaler.inverse_transform(ttfs)
# # ttfs = np.delete(ttfs, 0, 1)
# # predictions = np.clip(ttfs, 0, a_max=None)

# def convert_times(times):
#     # Convert to table with 1 column and len(times) rows.
#     times = np.expand_dims(times, 1)
#     # Add nonsense column to the left (for nonsense seismographic data)
#     times = np.append(np.zeros((len(times), 1)), times, axis=1)
#     # Invert the transformation
#     times = scaler.inverse_transform(times)
#     # Return transformed times.
#     return times[:,1]

# # def test_on_valid_set(n_tests):
#     # return scaler.inverse_transform(np.array([[0,model.evaluate_generator(gen, steps=n_steps)]]))[0][1]





# # ss = pd.read_csv(
# #     "data/sample_submission.csv",
# #     dtype = {
# #         # The seismic signal is int16, but we normalise it later on and
# #         # therefore we load the acoustic data as floats.
# #         "acoustic_data"   : np.float64,
# #         "time_to_failure" : np.float64 # The time (s) till next earthquake.
# #     })

# # for index, row in ss.iterrows():
# #     seg = row["seg_id"]

# #     valData = pd.read_csv(
# #         "data/test/" + seg + ".csv",
# #         dtype = {
# #             # The seismic signal is int16, but we normalise it later on and
# #             # therefore we load the acoustic data as floats.
# #             "acoustic_data"   : np.float64
# #         }
# #     )

# #     valData = np.append(valData.values, np.zeros((150000,1)), axis=1)
# #     valData = scaler.transform(valData)
# #     valData = np.delete(valData, 1, 1)

# #     valGen = generator(valData)

# #     loss = model.predict_generator(valGen, steps=100)

# #     break









# # # Visualize accuracies
# # import matplotlib.pyplot as plt

# # def perf_plot(history, what = 'loss'):
# #     x = history.history[what]
# #     val_x = history.history['val_' + what]
# #     epochs = np.asarray(history.epoch) + 1
    
# #     plt.plot(epochs, x, 'bo', label = "Training " + what)
# #     plt.plot(epochs, val_x, 'b', label = "Validation " + what)
# #     plt.title("Training and validation " + what)
# #     plt.xlabel("Epochs")
# #     plt.legend()
# #     plt.show()
# #     return None

# # perf_plot(history)

# # # Load submission file
# # submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# # # Load each test data, create the feature matrix, get numeric prediction
# # for i, seg_id in enumerate(submission.index):
# #   #  print(i)
# #     seg = pd.read_csv('../input/test/' + seg_id + '.csv')
# #     x = seg['acoustic_data'].values
# #     submission.time_to_failure[i] = model.predict(np.expand_dims(features(x), 0))

# # submission.head()

# # # Save
# # submission.to_csv('submission.csv')

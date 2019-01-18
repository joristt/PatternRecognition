import pandas as pd
import numpy  as np
import os
import matplotlib.pyplot as plt
from time import time

# In debug mode, only a small portion of the 629.145.481 rows from train.csv
# will be read.
DEBUG = True

# DATA
# https://www.kaggle.com/c/LANL-Earthquake-Prediction/data
# 
# train.csv contains 629.145.481 rows and is ~10GB
# 
data_dir = "./data/"
train_path = data_dir + "train.csv"

s = time()
print("Loading data into RAM...")
data = pd.read_csv(
	train_path,
	nrows=10e7 if DEBUG else None, # nrows=None causes all lines to be read.
	dtype = {
		"acoustic_data"   : np.int16,  # The seismic signal
		"time_to_failure" : np.float64 # The time (s) till next earthquake.

	}
).values # Convert pd.dataframe to np.ndarray

scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

print("Loaded data in %s seconds." % (time() - s))

# Calculate summary statistics per time step.
def summary_statistics(z):
	return np.c_[z.mean(axis=1), 
	             np.median(np.abs(z), axis=1),
	             z.std(axis=1), 
	             z.max(axis=1),
	             z.min(axis=1)]

# Calculates list of eartquake moments. The list contains tuples. Every tuple
# contains two indices. The first index is the last moment before the earth
# quake, the second index is always simply first index + 1, and is therefore
# the first moment after the earth quake.
def quake_moments(data):
	moms = np.where(data[1:,1] > data[:-1,1])[0]
	return zip(moms, moms+1)

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

# For a given ending position "u_bound", we split the last 150.000 values of "x" into 150 pieces of length 1000 each.
# From each piece, 16 features are extracted. This results in a feature matrix of dimension (150 time steps x 16 features). 
def features(data, u_bound=None, nr_subsets=150, subset_size=1000, mean=None, std=None):
    nr_values = nr_subsets * subset_size
    u_bound   = u_bound or len(data)
    l_bound   = u_bound - nr_values

    assert l_bound >= 0
    assert mean != None != std

    # Reshaping and standardize approximately
    l = (data[l_bound:u_bound].reshape(nr_subsets, -1) - mean) / std
    
    # Extracts features of sequences of full length 1000, of the last 100 values and finally also 
    # of the last 10 observations. 
    return np.c_[
    	summary_statistics(l),
    	summary_statistics(l[:, -subset_size // 10:]),
    	summary_statistics(l[:, -subset_size // 100:]),
    	l[:, -1:]
    ]

# We call "summary_statistics" three times, so the total number of features is 3 * 5 + 1 (last value) = 16
nr_features = 16 

# Calculate the mean and standard deviation of a sample of the acoustic values.
s           = sample(data[:,0])
sample_mean = s.mean()
sample_std  = s.std()


# The generator randomly selects "batch_size" ending positions of sub-time series. For each ending position,
# the "time_to_failure" serves as target, while the features are created by the function "features".
def generator(data):

    l_bound     = 0
    u_bound     = len(data) - 1
    batch_size  = 32
    n_steps     = 150
    step_length = 1000

    while True:
        # Pick indices of ending positions
        rows = np.random.randint(l_bound + n_steps * step_length, u_bound, size=batch_size)

        # Initialize feature matrices and targets
        samples = np.zeros((batch_size, n_steps, nr_features))
        targets = np.zeros(batch_size, )
        
        for j, row in enumerate(rows):
            samples[j] = features(
                data[:, 0],
                u_bound     = row,
                nr_subsets  = n_steps,
                subset_size = step_length,
                mean        = sample_mean,
                std         = sample_std
            )
            targets[j] = data[row, 1]
        yield samples, targets
        

train_gen = generator(data)
valid_gen = generator(data)

# Define model
# os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.models import Sequential
from keras.layers import Dense, CuDNNGRU, GRU
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint

# keras.layers.CuDNNGRU(units, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, return_sequences=False, return_state=False, stateful=False)
# keras.layers.      GRU(units, activation='tanh', recurrent_activation='hard_sigmoid', use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', bias_initializer='zeros', kernel_regularizer=None, recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, dropout=0.0, recurrent_dropout=0.0, implementation=1, return_sequences=False, return_state=False, go_backwards=False, stateful=False, unroll=False, reset_after=False)

cb = [ModelCheckpoint("model.hdf5", monitor='val_loss', save_weights_only=False, period=3)]

model = Sequential()
# NVIDIA Only: model.add(CuDNNGRU(48, input_shape=(None, nr_features)))
model.add(GRU(48, input_shape=(150, nr_features))) # (None, nr_features)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

model.summary()

# Compile and fit model
model.compile(optimizer=adam(lr=0.0005), loss="mae")

history = model.fit_generator(train_gen,
                              steps_per_epoch=1000,#n_train // batch_size,
                              epochs=10,
                              verbose=1,
                              callbacks=cb,
                              validation_data=valid_gen,
                              validation_steps=100)#n_valid // batch_size)

# # Visualize accuracies
# import matplotlib.pyplot as plt

# def perf_plot(history, what = 'loss'):
#     x = history.history[what]
#     val_x = history.history['val_' + what]
#     epochs = np.asarray(history.epoch) + 1
    
#     plt.plot(epochs, x, 'bo', label = "Training " + what)
#     plt.plot(epochs, val_x, 'b', label = "Validation " + what)
#     plt.title("Training and validation " + what)
#     plt.xlabel("Epochs")
#     plt.legend()
#     plt.show()
#     return None

# perf_plot(history)

# # Load submission file
# submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id', dtype={"time_to_failure": np.float32})

# # Load each test data, create the feature matrix, get numeric prediction
# for i, seg_id in enumerate(submission.index):
#   #  print(i)
#     seg = pd.read_csv('../input/test/' + seg_id + '.csv')
#     x = seg['acoustic_data'].values
#     submission.time_to_failure[i] = model.predict(np.expand_dims(features(x), 0))

# submission.head()

# # Save
# submission.to_csv('submission.csv')

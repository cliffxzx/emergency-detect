# Source: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

# %% Import libaries
import os

import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils, models, layers

# %% Define dataset's functions
# load a single file as a numpy array
def load_file(filepath) -> np.ndarray:
  dataframe = pd.read_csv(filepath, header=None, delim_whitespace=True)
  return dataframe.values

# load a list of files into a 3D array of [samples, timesteps, features]
def load_group(filenames, prefix=''):
  loaded = list()
  for name in filenames:
    data = load_file(prefix + name)
    loaded.append(data)
  # stack group so that features are the 3rd dimension
  loaded = np.dstack(loaded)
  return loaded

# load a dataset group, such as train or test
def load_dataset_group(group, prefix=''):
  filepath = prefix + group + '/Inertial Signals/'
  # load all 9 files as a single array
  filenames = list()
  # total acceleration
  # filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_'+group+'.txt', 'total_acc_z_'+group+'.txt']
  # body acceleration
  filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_'+group+'.txt', 'body_acc_z_'+group+'.txt']
  # body gyroscope
  filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_'+group+'.txt', 'body_gyro_z_'+group+'.txt']
  # load input data
  X = load_group(filenames, filepath)
  # load class output
  y = load_file(prefix + group + '/y_'+group+'.txt')
  return X, y

# load the dataset, returns train and test X and y elements
def load_dataset(path):
  # load all train
  trainX, trainy = load_dataset_group('train', path)
  print(trainX.shape, trainy.shape)
  # load all test
  testX, testy = load_dataset_group('test', path)
  print(testX.shape, testy.shape)
  # zero-offset class values
  trainy = trainy - 1
  testy = testy - 1
  # one hot encode y
  trainy = utils.to_categorical(trainy)
  testy = utils.to_categorical(testy)
  print(trainX.shape, trainy.shape, testX.shape, testy.shape)
  return trainX, trainy, testX, testy

# %% Define model's functions
# fit a model
def fit_model(trainX, trainy, epochs=10, batch_size=32, verbose=0):
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  model = models.Sequential([
    layers.Conv1D(filters=8, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
    layers.Conv1D(filters=8, kernel_size=3, activation='relu'),
    layers.Dropout(0.5),
    layers.MaxPooling1D(pool_size=2),
    layers.Flatten(),
    layers.Dense(25, activation='relu'),
    layers.Dense(n_outputs, activation='softmax'),
  ])

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # fit network
  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

  return model

# %%
if __name__ == '__main__':
  trainX, trainy, testX, testy = load_dataset('./dataset/uci_har_dataset/')
  # repeat experiment
  scores, model = [], None
  for r in range(1):
    sub_model = fit_model(trainX, trainy)
    _, score = sub_model.evaluate(testX, testy, batch_size=32, verbose=0)
    score = score * 100.0
    if all(score > x for x in scores):
      model = sub_model
    print('>#%d: %.3f' % (r+1, score))
    scores.append(score)

  model.summary()
  # summarize results
  print(scores)
  m, s = np.mean(scores), np.std(scores)
  print('Accuracy: %.3f%% (±%.3f)' % (m, s))

# %%
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

output_filename = 'emergency-detect'
open(f'output/{output_filename}.tflite', 'wb').write(tflite_model)
os.system(f'echo "#include \\"emergency-detect.h\\"\n" > output/{output_filename}.cc')
os.system(f'xxd -i output/{output_filename}.tflite >> output/{output_filename}.cc')

ops_details = tf.lite.Interpreter(model_path=f'output/{output_filename}.tflite')._get_ops_details()
ops_list = set(map(lambda x: x['op_name'], ops_details))
print("\nOperations list: ", ops_list)

# %%
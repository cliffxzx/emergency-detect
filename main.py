# Source: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

# %% Import libaries
import os

import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils, models, layers
from tensorflow.python.keras.layers.pooling import AveragePooling1D, GlobalAveragePooling1D

# %% Define dataset's functions
def load_dataset(path):
  x = pd.read_csv(f"{path}X_train.txt", header=None, delim_whitespace=True).values
  y = pd.read_csv(f"{path}y_train.txt", header=None, delim_whitespace=True).values
  x = x.reshape(x.shape[0], x.shape[1], 1)

  trainX, trainy = x[:int(len(x)*.8)], y[:int(len(x)*.8)]
  testX, testy = x[int(len(x)*.8):], y[int(len(x)*.8):]

  trainy = utils.to_categorical(trainy)
  testy = utils.to_categorical(testy)
  print(trainX.shape, trainy.shape, testX.shape, testy.shape)
  return trainX, trainy, testX, testy
# %% Define model's functions
# fit a model
def fit_model(trainX, trainy, epochs=10, batch_size=32, verbose=0):
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  model = models.Sequential([
    layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps, n_features)),
    layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.GlobalAveragePooling1D(),
    layers.Dense(n_outputs, activation='softmax'),
  ])

  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  # fit network
  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

  return model

# %%
if __name__ == '__main__':
  trainX, trainy, testX, testy = load_dataset('dataset/uci_har_dataset/train/')
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

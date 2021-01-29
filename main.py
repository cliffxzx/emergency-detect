# Source: https://machinelearningmastery.com/cnn-models-for-human-activity-recognition-time-series-classification/

# %% Import libaries
import os

import numpy as np
from numpy.lib.arraysetops import unique
import pandas as pd
import tensorflow as tf
from tensorflow.keras import utils, models, layers
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import AveragePooling1D, GlobalAveragePooling1D

# %% Define dataset's functions
def window(a, w, o, copy = False):
    sh = (a.size - w + 1, w)
    st = a.strides * 2
    view = np.lib.stride_tricks.as_strided(a, strides = st, shape = sh)[0::o]
    if copy:
        return view.copy()
    else:
        return view

labels = {'n': 0, 'b': 1, 'c': 2, 's': 3}

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def load_dataset(path):
  filenames = tf.io.gfile.glob(str(path) + '/*')
  x, y = np.empty(shape=(0,22050)), np.empty(shape=(0,1))
  for filename in filenames:
    if filename != 'dataset/joey_sound/info':
      lines = open(filename).readlines()
      sub_y = np.array([])
      for i in range(7):
        line = lines[i].split()
        sub_y = np.hstack((sub_y, np.full(int(line[2]) - int(line[1]), labels[line[0]])))

      sub_x = np.array(lines[7:], dtype='int')
      sub_x = window(sub_x, 22050, 128)
      sub_y = window(sub_y, 22050, 128)[:,-1::1]

      x = np.vstack((x, sub_x))
      y = np.vstack((y, sub_y))

  x = x.reshape(-1, 22050, 1);
  x, y = unison_shuffled_copies(x, y)

  trainX, trainy = x[:int(len(x)*.8)], y[:int(len(x)*.8)]
  testX, testy = x[int(len(x)*.8):], y[int(len(x)*.8):]
  trainy = utils.to_categorical(trainy)
  testy = utils.to_categorical(testy)
  print(trainX.shape, trainy.shape, testX.shape, testy.shape)
  return trainX, trainy, testX, testy
# %% Define model's functions
# fit a model
def fit_model(trainX, trainy, epochs=15, batch_size=32, verbose=1):
  n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
  model = models.Sequential([
    layers.Input(shape=(n_timesteps, n_features)),
    layers.AveragePooling1D(),
    layers.Conv1D(filters=8, kernel_size=256, activation='relu'),
    layers.MaxPooling1D(),
    layers.Conv1D(filters=16, kernel_size=256, activation='relu'),
    layers.GlobalMaxPooling1D(),
    layers.Dense(n_outputs, activation='softmax'),
  ])
  model.summary()
  opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
  model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  # fit network
  model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)

  return model

# %%
if __name__ == '__main__':
  with tf.device('/device:GPU:0'):
    trainX, trainy, testX, testy = load_dataset('dataset/joey_sound')
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

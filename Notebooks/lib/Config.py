import os
import pprint
import pandas as pd
import seaborn as sns
import tensorflow as tf
from pylab import rcParams

if 'COLAB_TPU_ADDR' not in os.environ:
  print('ERROR: Not connected to a TPU runtime.')
else:
  tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  print ('TPU address is', tpu_address)

  with tf.Session(tpu_address) as session:
    devices = session.list_devices()
    
  print('TPU devices:')
  pprint.pprint(devices)

try:
  # Deprecating
  tpu = tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
except ValueError:
  tpu = None

if tpu:
  tf.tpu.experimental.initialize_tpu_system(tpu)
  #strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128)
  #strategy = tf.contrib.distribute.TPUStrategy(tpu)
  strategy = tf.contrib.tpu.TPUDistributionStrategy(tpu)
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
  print('Strategy: ', strategy)
else:
  strategy = tf.distribute.get_strategy()
  print('Running on CPU')

def tfPredict(model, X_train, y_train, X_valid, y_valid, X_test, y_test):
  tpu_model = tf.contrib.tpu.keras_to_tpu_model(model,strategy)
  model.compile(optimizer=tf.train.AdamOptimizer(learning_rate),
                loss=tf.keras.losses.categorical_crossentropy,
                metrics=['binary_accuracy'])
  out = tpu_model.fit(X_train, y_train, 
                      epochs=10, batch_size = 1024, verbose=2,
                      validation_data=[X_valid, y_valid])
  
  print('\nHistory:', out.history)
  print('\nEvaluation on test data\n')
  results = model.evaluate(x_test, y_test, batch_size=1024)
  print('Test [loss, acc]:', results)

pd.set_option('display.max_rows', 500);
pd.set_option('display.max_columns', None);
rcParams['figure.figsize'] = 15, 8;
sns.set(style="whitegrid");

root_path = '/content/drive/My Drive/Colab Notebooks/'
myDrive_path = '/My Drive/Colab Notebooks/'
dataset_path = root_path + 'Dataset/'
dataset_preprocessed_path = root_path + 'PreprocessedDatasets/'
documents_path = root_path + 'Documents/'
resources_path = root_path + 'Resources/'
machineLearningModels_path = root_path + 'MachineLearningModels/'
notebooks_path = root_path + 'Notebooks/'
import pandas as pd
import seaborn as sns
import tensorflow as tf
import tensorflow.compat.v1 as tf
from pylab import rcParams

try:
  tf.disable_v2_behavior();
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver();
except ValueError:
  tpu = None

if tpu:
  tf.tpu.experimental.initialize_tpu_system(tpu);
  strategy = tf.distribute.experimental.TPUStrategy(tpu, steps_per_run=128)
  print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
else:
  strategy = tf.distribute.get_strategy()
  print('Running on CPU instead')
print("Number of accelerators: ", strategy.num_replicas_in_sync)

pd.set_option('display.max_rows', 500);
pd.set_option('display.max_columns', None);
rcParams['figure.figsize'] = 15, 8;
sns.set(style="whitegrid");

root_path = '/content/drive/My Drive/Colab Notebooks/'
myDrive_path = '/My Drive/Colab Notebooks/'
dataset_path = root_path + 'Dataset/'
dataset_preprocessed_path = root_path + 'Dataset_preprocessed/'
documents_path = root_path + 'Documents/'
resources_path = root_path + 'Resources/'
machineLearningModels_path = root_path + 'MachineLearningModels/'
notebooks_path = root_path + 'Notebooks/'
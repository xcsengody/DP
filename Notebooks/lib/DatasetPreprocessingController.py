import re
import os
import sys
import math
import socket
import json
import pylab as pl
import numpy as np
import geoip2.database
import matplotlib.pyplot as plt 
from pprint import pprint
from scipy import stats
from sklearn import preprocessing
from collections import Counter
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import (SMOTE,SVMSMOTE,ADASYN)
from sklearn.model_selection import (RandomizedSearchCV, GridSearchCV)

def saveDataset(dataset,fileName):
  dataset.to_csv(dataset_preprocessed_path+fileName+'.csv', index=False)
  print("Dataset saved to .../Colab Notebooks/PreprocessedDatasets/{}.csv".format(fileName))

def TreeBasedModelHyperparameterSelector(X_train, y_train)
  n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
  max_features = ['auto', 'sqrt', 'log2']
  max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
  max_depth.append(None)
  min_samples_split = [5, 10, 15, 20]
  min_samples_leaf = [2, 4, 6, 8, 10]
  bootstrap = [True, False]
  criterion = ['mse', 'mae']

  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap,
                'criterion': criterion}

  model = RandomForestRegressor()
  random_search = RandomizedSearchCV(estimator = model, param_distributions = random_grid, n_iter = 10, cv = 10, n_jobs=-1, verbose=2)
  randomForestRegressor_randomSearch = random_search.fit(X_train, y_train)
  params = randomForestRegressor_randomSearch.best_params_

  params_n_estimators = params['n_estimators']
  params_bootstrap = params['bootstrap']
  params_max_features = params['max_features']
  params_max_depth = params['max_depth']
  params_min_samples_split = params['min_samples_split']
  params_min_samples_leaf = params['min_samples_leaf']

  stop_val=10
  samples=4

  n_estimators = [int(x) for x in np.linspace(start = params_n_estimators, stop = params_n_estimators+stop_val, num = samples)]
  bootstrap = [params_bootstrap]
  max_features = [params_max_features]
  if params_max_depth is not None:
    max_depth = [int(x) for x in np.linspace(start= params_max_depth, stop= params_max_depth+stop_val, num = samples)]
  else:
    max_depth = [None]
  min_samples_split = [int(x) for x in np.linspace(start = params_min_samples_split, stop = params_min_samples_split+stop_val, num = samples)]
  min_samples_leaf = [int(x) for x in np.linspace(start = params_min_samples_leaf, stop = params_min_samples_leaf+stop_val, num = samples)]

  gridSearch_grid = {'n_estimators': n_estimators,
                    'bootstrap': bootstrap,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf}

  model = RandomForestRegressor()
  grid_search = GridSearchCV(model, param_grid = gridSearch_grid, cv = 3, n_jobs=-1, verbose = 2)
  randomForestRegressor_gridSearch = grid_search.fit(X_train, y_train)
  return randomForestRegressor_gridSearch.best_params_

def getProtocolNumber(x):
  try:
    return socket.getprotobyname(x);
  except:
    return -1;

#Ports range:
#Well known: 0-1023 (1)
#Registered: 1024-49151 (2)
#Private: 49152-65535 (3)

def portTypeNumeric(x):
  if x >= 0 and x < 1024:
    return 1;
  elif x >= 1024 and x < 49152:
    return 2;
  elif x >= 49152 and x < 65536:
    return 3;

def portTypeNominal(x):
  if x >= 0 and x < 1024:
    return "Well-Known";
  elif x >= 1024 and x < 49152:
    return "Registered";
  elif x >= 49152 and x < 65536:
    return "Private";

def isPrivateIP(x):
  #192.168.0.0 - 192.168.255.255
  #172.16.0.0 - 172.31.255.255
  #10.0.0.0 - 10.255.255.255
  if (re.search('192\.168\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])',x)) or (re.search('172\.16|31\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])',x)) or (re.search('10\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])',x)) or (re.search('127\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])',x)):
    return True
  else:
    return False

def isLocalhostIP(x):
  #127.0.0.1
  if (re.search('127\.0\.0\.1',x)):
    return True
  else:
    return False

def isMulticastIP(x):
  #224.0.0.0 - 239.255.255.255
  if (re.search('22[4-9]|23[0-9]\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])',x)):
    return True
  else:
    return False

def getIPLocation(reader, ip):
  try:
    location = reader.country(ip);
    #result = location.country.names['en']+" ("+location.country.iso_code+")";
    return location.country.iso_code;
  except:
    if isPrivateIP(ip):
      return "Private";
    elif isLocalhostIP(ip):
      return "Localhost";
    elif isMulticastIP(ip):
      return "Multicast"; 

def closeReader():
  reader.close();

def setPredictedValue(prediction):
  for p in prediction:
    value=p
    prediction.remove(p)
    return value

def makePlot(x,y,title,l):
  plt.figure();
  fig,ax = plt.subplots();
  mean_value=[np.mean(y)]*len(x);
  std_value=[np.std(y)]*len(x);
  data_line = ax.scatter(x, y, label='Data', marker='.', color="deepskyblue");
  mean_line = ax.plot(x, mean_value, label='Mean', linestyle='--', color="darkorange");
  std_line = ax.plot(x, std_value, label='STD', linestyle='-', color="olive");
  if l is not None: 
    line_value=[l]*len(x);
    threshold_line = ax.plot(x, line_value, label='Threshold', linestyle=':', color="red");
  legend = ax.legend(loc='upper right');
  plt.title(title);
  plt.show()

def makeMultiPlot(x1,y1,x2,y2,title):
  plt.figure();
  plt.scatter(x1, y1, label='Data_1', marker='o', color="deepskyblue");
  plt.scatter(x2, y2, label='Data_2', marker='o', color="darkorange");
  plt.legend(loc='upper right');
  plt.title(title);
  plt.show()

def plot_distribution(X, y, label='Classes'):
  colors = ['red','orange','lightgreen','darkgreen','lightblue','blue','yellow','pink','purple']
  markers = ['.']
  for l, c, m in zip(np.unique(y), colors, markers):
      plt.scatter(
          X[y==l, 0],
          X[y==l, 1],
          c=c, label=l, marker=m
      )
  plt.title(label)
  plt.legend(loc='upper right')
  plt.show()

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  
  from IPython.display import HTML
  import pandas as pd
  import numpy as np
  import base64
  csv = df.to_csv()
  b64 = base64.b64encode(csv.encode())
  payload = b64.decode()
  html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
  html = html.format(payload=payload,title=title,filename=filename)
  return HTML(html)

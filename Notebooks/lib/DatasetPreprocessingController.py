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
from sklearn.svm import LinearSVC
from imblearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from imblearn.over_sampling import (SMOTE,SVMSMOTE,ADASYN)
from sklearn.model_selection import ParameterGrid
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import (train_test_split,RandomizedSearchCV, GridSearchCV)
from sklearn.ensemble import ExtraTreesClassifier

def getProtocolNumber(x):
  try:
    return socket.getprotobyname(x);
  except:
    return -1;

def portType(x):
  #Ports range:
  #Well known: 0-1023 (1)
  #Registered: 1024-49151 (2)
  #Private: 49152-65535 (3)
  if x >= 0 and x < 1024:
    return 1;
  elif x >= 1024 and x < 49152:
    return 2;
  elif x >= 49152 and x < 65536:
    return 3;

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

def getIPLocation(ip):
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

def setPredictedValue():
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

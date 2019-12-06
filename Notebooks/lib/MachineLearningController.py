import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import statsmodels.api as sm
import parfit.parfit as pf

from google.colab import files
from collections import Counter
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import (SMOTE,SVMSMOTE,ADASYN)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn import linear_model
from sklearn.model_selection import ParameterGrid
from sklearn.kernel_approximation import Nystroem
from sklearn.model_selection import (train_test_split,RandomizedSearchCV, GridSearchCV)
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn import metrics
from pprint import pprint
from sklearn.metrics import (roc_curve, auc, roc_auc_score, accuracy_score, classification_report, confusion_matrix)
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib

datasets={}

def loadPreprocessedDatasets(datasets_to_load):
  datasets.clear()
  datasets_to_load_len = len(datasets_to_load)

  for d in datasets_to_load:
    df_name = d.split(".")[0]
    datasets[df_name] = pd.read_csv(dataset_preprocessed_path+d, delimiter=',', encoding='utf-8', low_memory=False, skipinitialspace=True, skip_blank_lines=True, verbose=False);

  print("Datasets:")
  for d in datasets:
    print(d)
  print("\nTo access dataset use datasets[\'datasetName\']\n")

def mergeDatasets(arrayOfDatasetsNames):
  frames = arrayOfDatasetsNames
  dataset = pd.concat(frames)
  return dataset.sample(frac=1).reset_index(drop=True)

def getDatasetDimensions(dataset):
  print("Rows: {}\nColumns: {}\n".format(dataset.shape[0],dataset.shape[1]))

def analyzeDataset(dataset):
  attack_traffic=dataset[dataset.label.apply(lambda x: x==1)]
  normal_traffic=dataset[dataset.label.apply(lambda x: x==0)]
  total_features=dataset.shape[1];
  total_rows=dataset.shape[0];
  normal_traffic_rows=normal_traffic.shape[0];
  normal_traffic_per=(100*normal_traffic_rows)/total_rows;
  attack_traffic_rows=attack_traffic.shape[0];
  attack_traffic_per=(100*attack_traffic_rows)/total_rows;

  print("Total rows: {}\nTotal features: {}\nNormal traffic: {} ({} %)\nAttack traffic: {} ({} %)\n".
        format(total_rows, total_features, normal_traffic_rows, float("{0:.2f}".format(normal_traffic_per)), attack_traffic_rows, float("{0:.2f}".format(attack_traffic_per))));

  count_classes = pd.value_counts(dataset.label)
  count_classes.plot(kind="bar")
  plt.title("Normal vs Attack class distibution")
  plt.xticks(range(2), ["Normal","Attack"])
  plt.xlabel("Class")
  plt.ylabel("Frequency")

def handleNaNcolumns():
  for d in datasets:
    dataset=datasets[d]
    print("{} - NaN columns: {}\n".format(d,dataset.columns[dataset.isna().any()].tolist()))

def splitDataset(dataset):
  y = dataset.label
  X = dataset.drop('label', axis=1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
  X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1)

  print("Train set size: {}\nValidation set size: {}\nTest set size: {}\n".format(X_train.shape[0],X_valid.shape[0],X_test.shape[0]))
  return X_train, y_train, X_valid, y_valid, X_test, y_test

def getClassWeights(y_train):
  class_weights_arr = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)
  class_weights = {};

  i=0
  for cw in class_weights_arr:
    class_weights[i] = cw;
    i+=1

  return class_weights

def handleMissingColumns(datasets, drop=False):
  dataset_names = list(datasets)
  dataset_names

  i=0
  while i < len(dataset_names):
    prev_df = dataset_names[i]
    next_df = dataset_names[i+1]

    df1 = datasets[str(prev_df)]
    df2 = datasets[str(next_df)]

    set1 = set(df1.columns)
    set2 = set(df2.columns)
    missing_cols_1 = list(sorted(set1 - set2))
    missing_cols_2 = list(sorted(set2 - set1))

    #if drop:
    #  if missing_cols_1:
    #    datasets[str(prev_df)].drop(columns=missing_cols_1)
    #  if missing_cols_2:
    #    datasets[str(next_df)].drop(columns=missing_cols_2)
    #else:
      #handle missing columns and its values
    i+=2

    #for j in list(df2):
    #  if j not in list(df1):
    #      print(j)

def loadModel(file):
  return joblib.load(machineLearningModels_path+file+'.sav')

def saveModel(dataset, fileName):
  joblib.dump(dataset, machineLearningModels_path+fileName+'.sav')
  print("Model saved to .../Colab Notebooks/MachineLearningModels/{}.sav\n".format(fileName))

def getCorelationMatrix(X_train):
  corrmat = X_train.corr()
  top_corr_features = corrmat.index
  pl.figure(figsize=(60,60))
  g=sns.heatmap(X_train[top_corr_features].corr(method='pearson', min_periods=1),annot=True,cmap="RdYlGn")

def plot_roc_curve(fpr, tpr):
  plt.plot(fpr, tpr, color='orange', label='ROC')
  plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Receiver Operating Characteristic (ROC) Curve')
  plt.legend()
  plt.show()

def predict(clf, X, y, title):
  print("\n")
  print(title)
  print("\n")
  prediction = clf.predict(X)

  accuracy = accuracy_score(y, prediction)
  print("Accuracy: %.2f%%\n" % (accuracy * 100.0))

  print("\nClassification Report\n {}\n".format(classification_report(y, prediction)))

  print("\nConfusion-matrix\n {}\n".format(pd.crosstab(y, prediction, rownames=['Actual Species'], colnames=['Predicted Species'])))

  proba = clf.predict_proba(X)
  proba = [p[1] for p in proba]
  print("\nROC-AUC: {}\n".format(roc_auc_score(y, proba)))

  fpr, tpr, thresholds = roc_curve(y.values, proba)
  plot_roc_curve(fpr, tpr)
  print("\n")

















  #score = metrics.f1_score(y_test, prediction, average=None)
  #print("F1 score: {}".format(score))

  #cv = cross_val_score(clf, X_train, y_train, cv=10, scoring='roc_auc')
  #print("Standard Cross-validation accuracy: %f (+/- %f)" % (cv.mean(), (cv.std()*2)))

  #skfold = StratifiedKFold(n_splits=10)

  #skfold_cv = cross_val_score(clf, X_train, y_train, cv=skfold, scoring='roc_auc')
  #print("Stratified K-fold Cross-validation accuracy of train set: %f (+/- %f)\n" % (skfold_cv.mean(), (skfold_cv.std()*2)))

  #skfold_cv = cross_val_score(clf, X_valid, y_valid, cv=skfold, scoring='roc_auc')
  #print("Stratified K-fold Cross-validation accuracy of validation set: %f (+/- %f)\n" % (skfold_cv.mean(), (skfold_cv.std()*2)))

  # Confusion-matrix usually used to evaluate the performance of a multiclass model.
  #conf_mat = confusion_matrix(y_test, prediction)
  #sns.heatmap(conf_mat,annot=True)
  #plt.title("Confusion-matrix")
  #plt.figure(figsize=(20,20))
  #plt.show()
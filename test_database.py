import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import plot_confusion_matrix

#Read 160,000 data points
#v3features_file = open(r"C:\Windows\System32\victoria\Training Set\V3features_embedded.pkl","rb")
#v3features_db = pickle.load(v3features_file)
#v3features_file.close()
#v3features_df = pd.DataFrame(v3features_db,columns=['x','y'])
#v3features = v3features_df.to_numpy()

# Read data from training set files and create dataframes
training_in_file = open(r"Sets\training_in_3.pkl","rb")
training_in_db = pickle.load(training_in_file)
training_in_file.close()
training_out_file = open(r"Sets\training_out_3.pkl","rb")
training_out_db = pickle.load(training_out_file)
training_out_file.close()
training_in_df = pd.DataFrame(training_in_db)
training_out_df = pd.DataFrame(training_out_db, columns=['label'])
training_in = training_in_df.to_numpy()
training_out_column = training_out_df.to_numpy()
training_out = np.ravel(training_out_column)

# Read data from testing set files and create dataframes
testing_in_file = open(r"Sets\testing_in_3.pkl","rb")
testing_in_db = pickle.load(testing_in_file)
testing_in_file.close()
testing_out_file = open(r"Sets\testing_out_3.pkl","rb")
testing_out_db = pickle.load(testing_out_file)
testing_out_file.close()
testing_in_df = pd.DataFrame(testing_in_db)
testing_out_df = pd.DataFrame(testing_out_db, columns=['label'])
testing_in = testing_in_df.to_numpy()
testing_out = testing_out_df.to_numpy()

# Fit SVM model
C=1.0
linsvc = svm.LinearSVC(C=C).fit(training_in, training_out)
svclink = svm.SVC(kernel='linear', C=C).fit(training_in, training_out)
svcrbfk = svm.SVC(kernel='rbf', C=C).fit(training_in, training_out)
svcpolyk = svm.SVC(kernel='poly', degree=3, C=C).fit(training_in, training_out)
# svcrbfk = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(training_in, training_out)

test = [15,9]
test = np.array(test)
test = test.reshape(1,-1)
print(linsvc.decision_function([[15,9]]))
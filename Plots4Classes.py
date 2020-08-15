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

# Read 160,000 data points
v3features_file = open(r"Data\V3features_embedded.pkl","rb")
v3features_db = pickle.load(v3features_file)
v3features_file.close()
v3features_df = pd.DataFrame(v3features_db,columns=['x','y'])
v3features = v3features_df.to_numpy()

# Read data from training set files and create dataframes
training_in_file = open(r"Sets\training_in_4.pkl","rb")
training_in_db = pickle.load(training_in_file)
training_in_file.close()
training_out_file = open(r"Sets\training_out_4.pkl","rb")
training_out_db = pickle.load(training_out_file)
training_out_file.close()
training_in_df = pd.DataFrame(training_in_db)
training_out_df = pd.DataFrame(training_out_db, columns=['label'])
training_in = training_in_df.to_numpy()
training_out_column = training_out_df.to_numpy()
training_out = np.ravel(training_out_column)

# Read data from testing set files and create dataframes
testing_in_file = open(r"Sets\testing_in_4.pkl","rb")
testing_in_db = pickle.load(testing_in_file)
testing_in_file.close()
testing_out_file = open(r"Sets\testing_out_4.pkl","rb")
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

# Read data for all 1000 points from file and create dataframe
labels_file = open(r"Data\labels.pkl","rb")
labels_db = pickle.load(labels_file)
labels_file.close()
labels_df = pd.DataFrame(labels_db)

#Graph correct labels
fig, ax = plt.subplots()
for i in range(len(testing_in_df)):
    ptcolor = ""
    ind = testing_in_df.index[i]
    if labels_df.loc[ind,'label'] == '1':
            ptcolor = "red"
    if labels_df.loc[ind,'label'] == '2':
        ptcolor = "limegreen"
    if labels_df.loc[ind,'label'] == '3':
        ptcolor = "gold"
    if labels_df.loc[ind,'label'] == '4':
        ptcolor = "coral"
    x = labels_df.loc[ind,'x']
    y = labels_df.loc[ind,'y']
    ax.scatter(x, y, color = ptcolor, s=25, alpha = 0.8)
ax.set_title('True Labels')

#Graph predicted labels - linear SVC
fig2, ax2 = plt.subplots()
for i in range(len(testing_in_df)):
    ptcolor = ""
    ind = testing_in_df.index[i]
    if linsvc.predict(testing_in)[i] == '1':
            ptcolor = "red"
    if linsvc.predict(testing_in)[i] == '2':
        ptcolor = "limegreen"
    if linsvc.predict(testing_in)[i] == '3':
        ptcolor = "gold"
    if linsvc.predict(testing_in)[i] == '4':
        ptcolor = "coral"
    x = v3features_df.iloc[ind,0]
    y = v3features_df.iloc[ind,1]
    ax2.scatter(x, y, color = ptcolor, s=25, alpha = 0.8)
    ax2.set_title('Predicted Labels: Linear SVC')

#Graph predicted labels - SVC with linear kernel
fig3, ax3 = plt.subplots()
for i in range(len(testing_in_df)):
    ptcolor = ""
    ind = testing_in_df.index[i]
    if svclink.predict(testing_in)[i] == '1':
            ptcolor = "red"
    if svclink.predict(testing_in)[i] == '2':
        ptcolor = "limegreen"
    if svclink.predict(testing_in)[i] == '3':
        ptcolor = "gold"
    if svclink.predict(testing_in)[i] == '4':
        ptcolor = "coral"
    x = v3features_df.iloc[ind,0]
    y = v3features_df.iloc[ind,1]
    ax3.scatter(x, y, color = ptcolor, s=25, alpha = 0.8)
    ax3.set_title('Predicted Labels: SVC with linear kernel')

#Graph predicted labels - SVC with RBF kernel
fig4, ax4 = plt.subplots()
for i in range(len(testing_in_df)):
    ptcolor = ""
    ind = testing_in_df.index[i]
    if svcrbfk.predict(testing_in)[i] == '1':
            ptcolor = "red"
    if svcrbfk.predict(testing_in)[i] == '2':
        ptcolor = "limegreen"
    if svcrbfk.predict(testing_in)[i] == '3':
        ptcolor = "gold"
    if svcrbfk.predict(testing_in)[i] == '4':
        ptcolor = "coral"
    x = v3features_df.iloc[ind,0]
    y = v3features_df.iloc[ind,1]
    ax4.scatter(x, y, color = ptcolor, s=25, alpha = 0.8)
    ax4.set_title('Predicted Labels: SVC with RBF kernel')

#Graph predicted labels - SVC with Polynomial kernel
fig5, ax5 = plt.subplots()
for i in range(len(testing_in_df)):
    ptcolor = ""
    ind = testing_in_df.index[i]
    if svcpolyk.predict(testing_in)[i] == '1':
            ptcolor = "red"
    if svcpolyk.predict(testing_in)[i] == '2':
        ptcolor = "limegreen"
    if svcpolyk.predict(testing_in)[i] == '3':
        ptcolor = "gold"
    if svcpolyk.predict(testing_in)[i] == '4':
        ptcolor = "coral"
    x = v3features_df.iloc[ind,0]
    y = v3features_df.iloc[ind,1]
    ax5.scatter(x, y, color = ptcolor, s=25, alpha = 0.8)
    ax5.set_title('Predicted Labels: SVC with Polynomial kernel')

# Create legend of colors corresponding to labels
spine_patch = mpatches.Patch(color = "red", label = "spine")
shaft_patch = mpatches.Patch(color = "limegreen", label = "shaft")
soma_patch = mpatches.Patch(color = "gold", label = "soma")
ais_patch = mpatches.Patch(color = "coral", label = "proximal process")
ax.legend(handles = [spine_patch, shaft_patch, soma_patch, ais_patch], fontsize = 'x-small')
ax2.legend(handles = [spine_patch, shaft_patch, soma_patch, ais_patch], fontsize = 'x-small')
ax3.legend(handles = [spine_patch, shaft_patch, soma_patch, ais_patch], fontsize = 'x-small')
ax4.legend(handles = [spine_patch, shaft_patch, soma_patch, ais_patch], fontsize = 'x-small')
ax5.legend(handles = [spine_patch, shaft_patch, soma_patch, ais_patch], fontsize = 'x-small')

plt.show()
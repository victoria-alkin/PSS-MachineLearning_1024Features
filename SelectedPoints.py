import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans

# Read data from full features file and create a dataframe
fullfeatures_file = open(r"Data\V3features.pkl","rb")
fullfeatures_db = pickle.load(fullfeatures_file)
fullfeatures_file.close()
fullfeatures_df = pd.DataFrame(fullfeatures_db)

# Read data from selected points file and create a dataframe
selectedpoints_file = open(r"Data\V3features_embedded_selected_files.pkl","rb")
selectedpoints_db = pickle.load(selectedpoints_file)
selectedpoints_file.close()
selectedpoints_df = pd.DataFrame(selectedpoints_db)
selectedpoints_df.to_csv("selectedpoints.csv")

# Make dataframe of only selected points with full data
full_selected_df = pd.DataFrame()
for i in selectedpoints_df.index:
    full_selected_df = full_selected_df.append(fullfeatures_df.iloc[i])

# Add labels column
labels_file = open(r"Data\labels.pkl","rb")
labels_db = pickle.load(labels_file)
labels_file.close()
labels_df = pd.DataFrame(labels_db)
full_selected_df["label"] = labels_df["label"]

print(full_selected_df)
full_selected_df.to_csv("full_selectedpoints.csv")
full_selected_file = open(r"full_selectedpoints.pkl","ab")
pickle.dump(full_selected_df, full_selected_file)
full_selected_file.close()
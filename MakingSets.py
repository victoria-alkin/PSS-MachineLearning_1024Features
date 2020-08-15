import numpy as np
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sklearn
from sklearn import metrics
from sklearn.cluster import KMeans

# Read data from full features selected points file and create a dataframe
full_selected_file = open(r"full_selectedpoints.pkl","rb")
full_selected_db = pickle.load(full_selected_file)
full_selected_file.close()
full_selected_df = pd.DataFrame(full_selected_db)
print(full_selected_df)

# Read data from file and create dataframe
labels_file = open(r"Data\labels.pkl","rb")
labels_db = pickle.load(labels_file)
labels_file.close()
labels_df = pd.DataFrame(labels_db,columns=['x','y','label'])
makeclusters_df = labels_df.drop(columns = 'label')
labels_df.to_csv(r"labels.csv")
makeclusters_df.to_csv(r"forkmeans.csv")

# Run KMeans on data to create 20 clusters
Kmean = KMeans(n_clusters=20)
Kmean.fit(makeclusters_df)
kmean_model_file = open(r"Kmean_sets_model.pkl","ab")
kmean_model_file = pickle.dumps(Kmean)

# Select 10 random points from each of 20 clusters for the testing set
testing = [0]*labels_df.shape[0] # list to hold 1s for indices of testing set points
for cluster_index in range(20):
    cluster_list = []

    for i in range(Kmean.labels_.size):
        if (Kmean.labels_[i]==cluster_index):
            cluster_list.append(i)
    testing_cluster_list = random.sample(cluster_list, 10)

    for i in testing_cluster_list:
        testing[i]=1

# Add column to specify training set points and write dataframe to csv and pickle files
full_selected_df['testing']=testing
full_selected_df.to_csv(r"full_trainingtestingpoints.csv")
full_selected_file = open(r"full_trainingtestingpoints.pkl","ab")
pickle.dump(full_selected_df, full_selected_file)
full_selected_file.close()

# Create new dataframe with only testing set points
full_testingpoints_df = pd.DataFrame()
for i in range(full_selected_df.shape[0]):
    if full_selected_df.iloc[i,1025] == 1:
        full_testingpoints_df = full_testingpoints_df.append(full_selected_df.iloc[i], ignore_index=False)

# Create new dataframe with only training set points
full_trainingpoints_df = pd.DataFrame()
for i in range(full_selected_df.shape[0]):
    if full_selected_df.iloc[i,1025] == 0:
        full_trainingpoints_df = full_trainingpoints_df.append(full_selected_df.iloc[i], ignore_index=False)

# Create csv and pickle files of testing points and training points
full_testingpoints_df.to_csv(r"full_testing.csv")
full_trainingpoints_df.to_csv(r"full_training.csv")

testing_file = open(r"full_testing.pkl","wb")
pickle.dump(full_testingpoints_df, testing_file)
testing_file.close()

training_file = open(r"full_training.pkl","wb")
pickle.dump(full_trainingpoints_df, training_file)
training_file.close()

# Plot selected points
#ax = testingpoints_df.plot(x='x',y='y',kind='scatter',color='red')
#trainingpoints_df.plot(x='x',y='y',kind='scatter',color='blue',ax=ax)

# Plot cluster centers
#cluster_centers_df = pd.DataFrame(Kmean.cluster_centers_,columns=['x','y'])
#cluster_centers_df.plot(x='x',y='y',kind='scatter',color='yellow', ax=ax)

# Create legend
#testing_patch = mpatches.Patch(color = "red", label = "testing points")
#training_patch = mpatches.Patch(color = "blue", label = "training points")
#centers_patch = mpatches.Patch(color = "yellow", label = "cluster centers")
#plt.legend(handles = [testing_patch, training_patch, centers_patch], fontsize = 'small', loc = 2)

#plt.show()
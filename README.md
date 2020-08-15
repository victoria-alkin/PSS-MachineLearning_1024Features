# PSS-MachineLearning_1024Features
Using machine learning models to classify post-synaptic shapes based on 1024-feature vector files created from mesh files by an autoencoder

MakingSets.py - creates pkl files with training and testing sets (selected using kmeans and random selection)

Sets.py - creates dataframes for training and testing input and output with 3 classes (spine, shaft, soma), 4 classes (spine, shaft, soma, proximal process), and 8 classes (spine, shaft, soma, proximal process, partial spine, partial shaft, merged spine, merged shaft)

3Classes.py, 4Classes.py, 8Classes.py - each python file trains four classifiers (linearSVC, SVM with linear kernel, SVM with RBF kernel, and SVM with polynomial kernel) and generates confusion matrices for all four. The 3Classes file uses only the spine, shaft, and soma classes. The 4 classes file also includes proximal processes. The 8 classes file also includes partial and merged spines and shafts.

Plots3Classes.py, Plots4Classes.py, Plots8Classes.py - each program creates graphs of the correct classifications of the testing set points as well as the predictions of all four classifiers

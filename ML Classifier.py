import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

data = load_breast_cancer()

label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']


print(label_names)
print(labels[0])
print(feature_names[0])
print(features[0])

train, test, train_labels, test_labels = train_test_split(features, labels, 
                                                          test_size=0.30,
                                                          random_state=4)

gnb = LinearDiscriminantAnalysis()

model = gnb.fit(train, train_labels)  #
preds = gnb.predict(test)             #.
print(preds)

scores = cross_val_score(gnb, train, train_labels, scoring='accuracy', cv = 5)

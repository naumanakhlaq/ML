import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import sklearn
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

path = os.getcwd() + '\\US Census 2015.csv'  
dataset = pd.read_csv(path)

features = dataset.values[:, 3:37]
labels = dataset.values[:, 16]


train, test, train_labels, test_labels = train_test_split(features, labels, 
                                                          test_size=0.30,
                                                          random_state=4)

mod = LinearRegression()
mod.fit(train, train_labels)

'''
preds = mod.predict(test)             
print(preds)

dataset.corr()
'''

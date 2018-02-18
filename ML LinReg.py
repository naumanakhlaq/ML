
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets as skd
from lxml import html
import requests

# 5, 6
v1 = 8
v2 = 9

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv'
#names = ['Frequency', 'Angle of attack', 'Chord length', 'Free-stream velocity', 'SSDT', 'Scaled Sound Pressure']
dataset = pd.read_csv(url)

array = np.array(dataset.values)

X = array[:, v1]
Y = array[:, v2]
#plt.ylim(ymin=0, ymax=50)

Xi = np.array(X, dtype=float)
Yi = np.array(Y, dtype=float)

plt.figure(figsize=(9.8, 6))
plt.scatter(Xi, Yi)

plt.plot(np.unique(Xi), np.poly1d(np.polyfit(Xi, Yi, 1))(np.unique(Xi))) #

mod = LinearRegression()
#train, test, train_labels, test_labels = train_test_split(Xi, Yi, 
#                                                          test_size=0.30,
#                                                          random_state=4)

X = dataset['temp']
Y = dataset['RH']

#model = mod.fit(X, Y)
model.get_params()
print('1')

#scores1 = cross_val_score(mod, test, test_labels, scoring='accuracy', cv = 5)

ax = list(dataset)

plt.xlabel(ax[v1]) 
plt.ylabel(ax[v2])

plt.show()


'''

names = np.array(names)

a = numpy.asarray(names)
cwd = os.getcwd()
numpy.savetxt(cwd+"dataa.csv", a, delimiter=",""")'''
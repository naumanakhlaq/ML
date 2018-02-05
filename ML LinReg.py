
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
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
dataset = pd.read_csv(url, delim_whitespace = False)

array = dataset.values
X = array[:, v1]
Y = array[:, v2]
#plt.ylim(ymin=0, ymax=50)

Xi = np.array(X, dtype=float)
Yi = np.array(Y, dtype=float)

plt.figure(figsize=(9.8, 6))
plt.scatter(Xi, Yi)
plt.plot(np.unique(Xi), np.poly1d(np.polyfit(Xi, Yi, 1))(np.unique(Xi))) #now try ML packages


ax = list(dataset)

plt.xlabel(ax[v1]) 
plt.ylabel(ax[v2])

plt.show()


'''

names = np.array(names)

a = numpy.asarray(names)
cwd = os.getcwd()
numpy.savetxt(cwd+"dataa.csv", a, delimiter=",""")'''
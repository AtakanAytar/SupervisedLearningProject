from enum import auto
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

import os
import tarfile
import urllib
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split

df_ksi = pd.read_csv ("./KSI.csv")


print("First three records")
print(df_ksi.head(3))



print("Statistics")
print(df_ksi.describe())

print("Dimentions")
print(df_ksi.shape)
print("Types")
print(df_ksi.dtypes)

print("Col Names")
print(df_ksi.columns.values)

print("Non null count")
print(df_ksi.info())

#See coords of the accidents
df_ksi.plot(kind="scatter", x="X", y="Y")
plt.title("Coords of Accidents")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#Frequency histograms of important columns
df_ksi["DISTRICT"].hist(bins=30, figsize=(25,15))
plt.xlabel("DISTRICT")
plt.ylabel("freqeuncy")
plt.show()

df_ksi["LIGHT"].hist(bins=30, figsize=(25,15))
plt.xlabel("LIGHT")
plt.ylabel("freqeuncy")
plt.show()

df_ksi["ALCOHOL"].hist(bins=30, figsize=(25,15))
plt.xlabel("ALCOHOL")
plt.ylabel("freqeuncy")
plt.show()

df_ksi["VEHTYPE"].hist(bins=40, figsize=(25,15))
plt.xlabel("VEHTYPE")
plt.ylabel("freqeuncy")
plt.show()

#See pairplot
sns.pairplot(df_ksi)
plt.show()

#histogram of all the numerical columns
df_ksi.hist(bins=50, figsize=(20,15))
plt.show()



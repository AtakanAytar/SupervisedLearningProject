#%%
from os import path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%
# LOADING DATA
data_filename = path.join('.', "KSI.csv")
output_dir = path.join('.', "output")
df_ksi = pd.read_csv(data_filename)
#%%

print("First three records")
print(df_ksi.head(3))
#%%


print("Statistics")
print(df_ksi.describe())
#%%

print("Dimentions")
print(df_ksi.shape)
print("Types")
print(df_ksi.dtypes)

print("Col Names")
print(df_ksi.columns.values)

print("Non null count")
print(df_ksi.info())

#%%

df_ksi["ocean_proximity"].unique()
df_ksi["ocean_proximity"].value_counts()

#%%

#See coords of the accidents
df_ksi.plot(kind="scatter", x="X", y="Y", alpha=.1)
plt.title("Coords of Accidents")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
plt.savefig(path.join(output_dir, 'scatter-coords-of-accidents.png'))

#%%

#Frequency histograms of important columns
df_ksi["DISTRICT"].hist(bins=30, figsize=(25,15))
plt.xlabel("DISTRICT")
plt.ylabel("freqeuncy")
plt.show()

#%%

df_ksi["LIGHT"].hist(bins=30, figsize=(25,15))
plt.xlabel("LIGHT")
plt.ylabel("freqeuncy")
plt.show()
#%%

df_ksi["ALCOHOL"].hist(bins=30, figsize=(25,15))
plt.xlabel("ALCOHOL")
plt.ylabel("freqeuncy")
plt.show()
#%%

df_ksi["VEHTYPE"].hist(bins=40, figsize=(25,15))
plt.xlabel("VEHTYPE")
plt.ylabel("freqeuncy")
plt.show()
#%%

#See pairplot
# sns.pairplot(df_ksi)
# plt.show()

#%%

#histogram of all the numerical columns
df_ksi.hist(bins=50, figsize=(20,15))
plt.show()

#%%


# CHECK FOR MISSING VALUES
missing_rows = df_ksi[df_ksi.isnull().any(axis=1)]
print (len(missing_rows))
#%%
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



# initial feature selection (tentative)
columns_selected = ["HOUR", "TIME", "STREET1", "DISTRICT", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "INVTYPE", "INVAGE", "VEHTYPE", "ACCLASS"]

# output column
output_column = "ACCLASS"
dfksi_main = df_ksi[columns_selected]
dfksi_main.info()

# recheck the missing values
dfksi_main.isnull().sum()

# any rows with missing values left with features selected
dfksi_main[dfksi_main.isnull().any(axis=1)]

# since there are relatively low data with missing values, we can delete them for now
# later we want to check for the countinous data and fill them with mean value
dfksi_main = dfksi_main.dropna()

# no more missing values
dfksi_main.isnull().sum()


#Data Split
from sklearn.model_selection import StratifiedShuffleSplit

# random seed
RAND_SEED = 123
SPLIT_SIZE = 0.2  # 20% test
splitter = StratifiedShuffleSplit(n_splits=1, test_size=SPLIT_SIZE, random_state=RAND_SEED)

print("Data shape:", dfksi_main.shape)
dfksi_main_X = dfksi_main.drop(output_column, axis=1)  # X
dfksi_main_y = dfksi_main[output_column]  # y

print("features:", dfksi_main_X.shape)
print("output:", dfksi_main_y.shape)


for train_index, test_index in splitter.split(dfksi_main_X, dfksi_main_y):
    dfksi_train_X = dfksi_main_X.iloc[train_index]
    dfksi_train_y = dfksi_main_y.iloc[train_index]
    dfksi_test_X = dfksi_main_X.iloc[test_index]
    dfksi_test_y = dfksi_main_y.iloc[test_index]
    
print("train X shape:", dfksi_train_X.shape)
print("train y shape:", dfksi_train_y.shape)
print("test X shape:", dfksi_test_X.shape)
print("test y shape:", dfksi_test_y.shape)

dfksi_training = dfksi_train_X.copy()
dfksi_training['ACCLASS'] = dfksi_train_y.copy()
dfksi_training.head(3)


#Pipeline

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# checking for categorical columns
# we have 2 numerical and 10 categorical data
dfksi_training.info()

dfksi_numerical_columns = ['HOUR', 'TIME']
dfksi_categorical_columns = ['STREET1', 'DISTRICT', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'VEHTYPE']


# create separate transformer pipeline for numerica and categorical values
numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)

pre_processor = ColumnTransformer(
    transformers=[
        ("numeric", numerical_pipeline, dfksi_numerical_columns),
        ("categoric",  OneHotEncoder(handle_unknown="ignore"), dfksi_categorical_columns),
    ]
)


#The Model

from sklearn.linear_model import LogisticRegression

pipeline = Pipeline(
    steps=[("preprocessor", pre_processor), ("classifier", LogisticRegression())]
)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

pipeline.fit(dfksi_train_X, dfksi_train_y)
print("model score: %.3f" % pipeline.score(dfksi_test_X, dfksi_test_y))

## Decison Trees Model 
## Suraj Regmi
from sklearn.tree import DecisionTreeClassifier

pipelineForHT = Pipeline(
    steps=[("preprocessor", pre_processor), 
           ("classifier", DecisionTreeClassifier(random_state=39))]
)

## hyper parameter tunining 
param_grid = {
    "classifier__criterion":["gini","entropy"],
    "classifier__max_depth":[2,4,8,10,12,14,18,20],
    }

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipelineForHT, param_grid)
grid.fit(dfksi_train_X, dfksi_train_y)

grid.best_score_
grid.best_params_
grid.cv_results_


pipeline3 = Pipeline(
    steps=[("preprocessor", pre_processor), 
           ("classifier", DecisionTreeClassifier(criterion="entropy",max_depth=20,random_state=39))]
)
pipeline3.fit(dfksi_train_X, dfksi_train_y)
print("Decision Tree model score: %.3f" % pipeline3.score(dfksi_test_X, dfksi_test_y))




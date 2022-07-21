#%%
# from enum import auto
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# import pandas as pd
# import seaborn as sns
# from sklearn.model_selection import train_test_split

from os import path
import numpy as np
import pandas as pd
# from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit

#%%
# DATA LOAD
data_filename = path.join('.', "KSI.csv")
df_ksi = pd.read_csv (data_filename)

#%%

## DATA EXPLORATION
# initial data exploration
print("First three records")
print(df_ksi.head(3))
#%%
print("Statistics")
print(df_ksi.describe())
#%%

print("Dimensions: ", df_ksi.shape)
print()
print("Types")
print(df_ksi.dtypes)
print("Col Names")
print(df_ksi.columns.values)
print("Non null count")
#%%
print(df_ksi.info())
#%%

# INITIAL VISUALIZATION
########################
# later we will sort out the graphs and use subplots
# 


# See coords of the accidents
df_ksi.plot(kind="scatter", x="X", y="Y", alpha=0.1)
plt.title("Coords of Accidents")
plt.xlabel("Lattitude")
plt.ylabel("Longitude")
plt.show()

#%%

# Frequency histograms of important columns
df_ksi["DISTRICT"].hist(bins=30, figsize=(25,15))
plt.xlabel("DISTRICT")
plt.ylabel("freqeuncy")
plt.show()

#%%

df_ksi["ROAD_CLASS"].hist(bins=30, figsize=(25,15))
plt.xlabel("ROAD_CLASS")
plt.ylabel("freqeuncy")
plt.show()

#%%

df_ksi["ACCLOC"].hist(bins=30, figsize=(25,15))
plt.xlabel("ACCLOC")
plt.ylabel("freqeuncy")
plt.show()

#%%
df_ksi["TRAFFCTL"].hist(bins=30, figsize=(25,15))
plt.xlabel("TRAFFCTL")
plt.ylabel("freqeuncy")
plt.show()

#%%

df_ksi["VISIBILITY"].hist(bins=30, figsize=(25,15))
plt.xlabel("VISIBILITY")
plt.ylabel("freqeuncy")
plt.show()

#%%

df_ksi["LIGHT"].hist(bins=30, figsize=(25,15))
plt.xlabel("LIGHT")
plt.ylabel("freqeuncy")
plt.show()

#%%

df_ksi["RDSFCOND"].hist(bins=30, figsize=(25,15))
plt.xlabel("RDSFCOND")
plt.ylabel("freqeuncy")
plt.show()

df_ksi["ALCOHOL"].hist(bins=30, figsize=(25,15))
plt.xlabel("ALCOHOL")
plt.ylabel("freqeuncy")
plt.show()

df_ksi["ACCLASS"].hist(bins=30, figsize=(25,15))
plt.xlabel("ACCLASS")
plt.ylabel("freqeuncy")
plt.show()

df_ksi["IMPACTYPE"].hist(bins=40, figsize=(25,15))
plt.xlabel("IMPACTYPE")
plt.ylabel("freqeuncy")
plt.show()

#%%

ax= plt.subplot()
df_ksi["INVTYPE"].hist(bins=30, figsize=(25,15))
plt.xlabel("INVTYPE")
plt.ylabel("freqeuncy")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()

#%%

df_ksi["INVAGE"].hist(bins=50, figsize=(25,15))
plt.xlabel("INVAGE")
plt.ylabel("freqeuncy")
plt.show()

ax= plt.subplot()
df_ksi["VEHTYPE"].hist(bins=40, figsize=(25,15))
plt.xlabel("VEHTYPE")
plt.ylabel("freqeuncy")
plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')
plt.show()
#%%
# See pairplot
sns.pairplot(df_ksi)
plt.show()

#%%

# histogram of all the numerical columns
df_ksi.hist(bins=50, figsize=(20,15))
plt.show()

#%%

# Pie chart
fig1, ax1 = plt.subplots()
explode = [0,0.1,0]
df_ksi.groupby('ACCLASS').size().plot(kind='pie', explode=explode, autopct='%.2f%%', ax=ax1)
ax1.set_ylabel('Accident Class', size=16)
ax1.axis('equal') 
plt.show()

#%%

fig1, ax1 = plt.subplots()
df_ksi.groupby('IMPACTYPE').size().plot(kind='pie', autopct='%.2f%%', ax=ax1)
ax1.set_ylabel('Impact Type', size=22)
ax1.axis('equal') 
plt.show()

#%%

fig1, ax1 = plt.subplots()
df_ksi.groupby('DISTRICT').size().plot(kind='pie', autopct='%.2f%%', ax=ax1)
ax1.set_ylabel('District', size=16)
ax1.axis('equal') 
plt.show()

#%%

fig1, ax1 = plt.subplots()
df_ksi.groupby('ACCLOC').size().plot(kind='pie', autopct='%.2f%%', ax=ax1)
ax1.set_ylabel('Collision Location', size=16)
ax1.axis('equal') 
plt.show()

#%%

fig1, ax1 = plt.subplots()
df_ksi.groupby('TRAFFCTL').size().plot(kind='pie', autopct='%.2f%%', ax=ax1, figsize=(20,15))
ax1.set_ylabel('Traffic Control Type', size=16)
ax1.axis('equal') 
plt.show()
## End of Data exploration

#%%
#@note
pd.crosstab(df_ksi["VEHTYPE"], df_ksi["ACCLASS"])
#%%
pd.crosstab(df_ksi["INJURY"], df_ksi["ACCLASS"])

#%%
## FEATURE SELECTION
#####################

df_ksi.columns.values
#%%

# initial feature selection (tentative)
columns_selected = ["HOUR", "TIME", "STREET1", "STREET2", "DISTRICT", "HOOD_ID", "TRAFFCTL", "VISIBILITY", "LIGHT", "RDSFCOND", "IMPACTYPE", "INVTYPE", "INVAGE", "VEHTYPE","LONGITUDE", "LATITUDE",  "ACCLASS"]

# output column
output_column = "ACCLASS"
#%%
dfksi_main = df_ksi[columns_selected]
dfksi_main.head(3)
#%%
dfksi_main.info()

#%%
# recheck the missing values
df_ksi.isnull().sum()
df_ksi.replace('<Null>', np.nan, inplace=True)

#%%
df_ksi.isnull().sum()

#%%

# some testing
df_ksi["IMPACTYPE"].unique()
df_ksi["IMPACTYPE"].value_counts()

#%%


# any rows with missing values left with features selected
df_ksi[df_ksi.isnull().any(axis=1)]

#%%

# since there are relatively low data with missing values, we can delete them for now
# later we want to check for the countinous data and fill them with mean value
# dfksi_main = dfksi_main.dropna()


#%%


# DATA SPLIT
############
from sklearn.model_selection import StratifiedShuffleSplit

RAND_SEED = 123
SPLIT_SIZE = 0.2  # 20% test

# using stratified shuffle split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=SPLIT_SIZE, random_state=RAND_SEED)
#%%
# separating the feature and target/output
print("Data shape:", dfksi_main.shape)
dfksi_main_X = dfksi_main.drop(output_column, axis=1)  # X
dfksi_main_y = dfksi_main[output_column]  # y
print("features:", dfksi_main_X.shape)
print("output:", dfksi_main_y.shape)

#%%
# splitting the data in training/testing
for train_index, test_index in splitter.split(dfksi_main_X, dfksi_main_y):
    dfksi_train_X = dfksi_main_X.iloc[train_index]
    dfksi_train_y = dfksi_main_y.iloc[train_index]
    dfksi_test_X = dfksi_main_X.iloc[test_index]
    dfksi_test_y = dfksi_main_y.iloc[test_index]

# final split result
print("train X shape:", dfksi_train_X.shape)
print("train y shape:", dfksi_train_y.shape)
print("test X shape:", dfksi_test_X.shape)
print("test y shape:", dfksi_test_y.shape)

#%%

# BUILDING THE PIPELINE
#######################

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

#%%
dfksi_main.info()

#%%
# checking for categorical columns
# we have 2 numerical and 10 categorical data
dfksi_numerical_columns = ['HOUR', 'TIME', 'HOOD_ID', 'LATITUDE', 'LONGITUDE']
dfksi_categorical_columns = ['STREET1', 'STREET2', 'DISTRICT', 'TRAFFCTL', 'VISIBILITY', 'LIGHT',
       'RDSFCOND', 'IMPACTYPE', 'INVTYPE', 'INVAGE', 'VEHTYPE']

#%%


df_ksi_num=df_ksi.select_dtypes(include='number')
df_ksi_cat=dfksi_main.select_dtypes(include='object')
df_ksi_cat.columns
#%%
# create separate transformer pipeline for numerica and categorical values
numerical_pipeline = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
)
numerical_pipeline

#%%
catgorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown="ignore")),
    ])
catgorical_pipeline

#%%
# creating column transformer with numeric and categorical transformers
pre_processor = ColumnTransformer(
    transformers=[
        ("numeric", numerical_pipeline, dfksi_numerical_columns),
        ("categoric",  catgorical_pipeline, dfksi_categorical_columns),
    ]
)
pre_processor

#%%

# dfksi_main = full_pipeline.fit_transform(housing)


#%%

## BUILDING THE MODELS
######################

## LOGISTIC REGRESSION
# Alamin Ahmed
# building and testing the logistic regression model
from sklearn.linear_model import LogisticRegression

# pipeline including the pre-processor
pipeline = Pipeline(
    steps=[("preprocessor", pre_processor), ("classifier", LogisticRegression(max_iter=200))]
)

#%%
# fitting the model
pipeline.fit(dfksi_train_X, dfksi_train_y)
# printing initial model score
print("model score: %.3f" % pipeline.score(dfksi_test_X, dfksi_test_y))

#%%
## DECISION TREE 
## Suraj Regmi
from sklearn.tree import DecisionTreeClassifier

# pipeline including the pre-processor
pipelineForHT = Pipeline(
    steps=[("preprocessor", pre_processor), 
           ("classifier", DecisionTreeClassifier(random_state=39))]
)

#%%
## hyper parameter tunining 
param_grid = {
    "classifier__criterion":["gini","entropy"],
    "classifier__max_depth":[2,4,8,10,12,14,18,20],
    }

# using Grid search for fine tuning
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipelineForHT, param_grid)
grid.fit(dfksi_train_X, dfksi_train_y)

# checking the scores
grid.best_score_
grid.best_params_
grid.cv_results_

#%%

pipeline3 = Pipeline(
    steps=[("preprocessor", pre_processor), 
           ("classifier", DecisionTreeClassifier(criterion="entropy",max_depth=20,random_state=39))]
)
pipeline3.fit(dfksi_train_X, dfksi_train_y)
print("Decision Tree model score: %.3f" % pipeline3.score(dfksi_test_X, dfksi_test_y))

#%%


## SUPPORT VECTOR
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
#%%

for k in ['linear', 'rbf', 'poly', 'sigmoid']:
    pipeline2 = Pipeline(
        steps=[("preprocessor", pre_processor), 
               ("classifier", SVC(kernel=k, gamma='auto'))]
    )
    pipeline2.fit(dfksi_train_X, dfksi_train_y)
    scores = cross_val_score(pipeline2, dfksi_train_X, dfksi_train_y, scoring='accuracy', cv=5)
    print(k)
    print ('Min:')
    print (scores.min())
    print ('Mean:')
    print (scores.mean())
    print ('Max:')
    print (scores.max())
    print("SVM model score: %.3f" % pipeline2.score(dfksi_test_X, dfksi_test_y))
    print('\n')

#%%


##Random Forest Regression 
#Atakan Aytar

#%%

from sklearn.ensemble import RandomForestClassifier
# Commented  out since it takes a long time to run Cross validation and started giving errors but ran once before that

# param_grid = { 
#     'n_estimators': [100, 200],
#      criterion{“gini”, “entropy”, “log_loss”}
#     'max_depth' : [4,5,6,None],
# }

# pipelineForRF = Pipeline(
    # steps=[("preprocessor", pre_processor), 
    #        ("classifier", RandomForestClassifier(random_state = 42))]
# )
# from sklearn.model_selection import GridSearchCV
# grid2 = GridSearchCV(pipelineForRF, param_grid)
# grid2.fit(dfksi_train_X, dfksi_train_y)
# print(grid2.best_params_)

pipeline4 = Pipeline(
    steps=[("preprocessor", pre_processor), 
           ("classifier", RandomForestClassifier(n_estimators = 100, random_state = 42,max_depth=None,criterion="gini"))]
)

#%%
pipeline4.fit(dfksi_train_X, dfksi_train_y)
print("Random forest model score: %.3f" % pipeline4.score(dfksi_test_X, dfksi_test_y))

#%%
#End of random forest

#Neural Network Model - Jamaal Bernabe
#import MLP Classifier
print("Neural Network Model")
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=123)
pipe = Pipeline(steps=[('pre',pre_processor),('mlpc', mlp)])

#GridSearch
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
#Function to display metrics easily
def scoremodel(model):
    print("Training Accuracy: " , model.score(dfksi_train_X,dfksi_train_y))
    print("Test Accuracy: " , model.score(dfksi_test_X,dfksi_test_y))
    y_pred = model.predict(dfksi_test_X)
    print("Accuracy Matrix: " , confusion_matrix(dfksi_test_y, y_pred))
    print("Best Parameters: " ,model.best_params_)
    print("Best Score: " ,model.best_score_)
    
param_grid = {
            'mlpc__max_iter': [500,600],
            'mlpc__hidden_layer_sizes': [(8,4,2),(4,2,1)],
            'mlpc__activation': ['tanh', 'relu'],
            'mlpc__solver': ['sgd', 'adam'],
            'mlpc__alpha': [0.0001,0.001, 0.01],
            'mlpc__learning_rate': ['constant','adaptive'],
}

rnd_grid = RandomizedSearchCV(estimator=pipe, param_distributions= param_grid, cv=3,n_jobs = -1,n_iter=10)
rnd_result = rnd_grid.fit(dfksi_train_X,dfksi_train_y)
scoremodel(rnd_grid)

best_params = [
    {'mlpc__activation': 'tanh', 'mlpc__alpha': 0.01, 'mlpc__hidden_layer_sizes': (8, 4, 2), 'mlpc__learning_rate': 'constant', 'mlpc__max_iter': 600, 'mlpc__solver': 'adam'},
    {'mlpc__activation': 'tanh', 'mlpc__alpha': 0.01, 'mlpc__hidden_layer_sizes': (4, 2, 1), 'mlpc__learning_rate': 'constant', 'mlpc__max_iter': 600, 'mlpc__solver': 'adam'},
    {'mlpc__activation': 'tanh', 'mlpc__alpha': 0.01, 'mlpc__hidden_layer_sizes': (8, 4, 2), 'mlpc__learning_rate': 'adaptive', 'mlpc__max_iter': 600, 'mlpc__solver': 'adam'},
    {'mlpc__activation': 'tanh', 'mlpc__alpha': 0.01, 'mlpc__hidden_layer_sizes': (4, 2, 1), 'mlpc__learning_rate': 'adaptive', 'mlpc__max_iter': 600, 'mlpc__solver': 'adam'}
    ]
#5 minutes
grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, n_jobs = -1,scoring="accuracy",refit=True)
grid_result = grid.fit(dfksi_train_X,dfksi_train_y)
y_pred = grid_result.predict(dfksi_test_X)
scoremodel(grid_result)

#Graph
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics

ConfusionMatrixDisplay.from_estimator(grid_result,dfksi_train_X,dfksi_train_y)
ConfusionMatrixDisplay.from_predictions(y_pred,dfksi_test_y)

fpr, tpr, thresholds = metrics.roc_curve(y_pred,dfksi_test_y, y_pred)
roc_auc = metrics.auc(fpr, tpr)
display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                               estimator_name='Neural Network')
display.plot()
plt.show()
# %%

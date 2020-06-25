import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# import random forest model
from sklearn.ensemble import RandomForestRegressor

# import things for cross validation
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

# import evaluation metrics 
from sklearn.metrics import mean_squared_error, r2_score

# import module for saving sci-kit learn models
from sklearn.externals import joblib


data_url = "http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
wine_data = pd.read_csv(data_url, sep=';')

# split the data set 
X = wine_data.iloc[:, :-1]
y = wine_data.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123, stratify=y)
y_train = y_train.to_numpy().flatten()
y_test = y_test.to_numpy().flatten()

# create pipeline with the preprocessing step and the model 
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# create hyperparameters - parameters for the Random Forest function
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# GridSearchCV takes in our model and tests out all of the possible parameters (using K-folds) that 
# we inputted so that we can get the best possible combo
clf = GridSearchCV(pipeline, hyperparameters, cv=10)

clf.fit(X_train, y_train)

# save our model
joblib.dump(clf, 'rf_regressor.pkl')

# predict results from testing data
clf = joblib.load('rf_regressor.pkl')
y_pred = clf.predict(X_test)

print(mean_squared_error(y_test, y_pred))

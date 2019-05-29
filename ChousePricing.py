# -*- coding: utf-8 -*-
"""
Created on Wed May 22 16:57:29 2019

@author: adam
"""

import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

#Convert data frame object to numpy array with features selection
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
        rooms_per_family = X[:, rooms_ix] / X[:, household_ix]
        population_per_family = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_rooms = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_family, population_per_family, bedrooms_per_rooms]
        else:
            return np.c_[X, rooms_per_family, population_per_family]
        
#Label binarizer do not fit the interface (only 2 positional arguments are allowed)
class CustomLabelBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, sparse_output=False):
        self.sparse_output = sparse_output
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        enc = LabelBinarizer(sparse_output=self.sparse_output)
        return enc.fit_transform(X)

#Import housing price data
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("data sets", "flats")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
#Load data
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

def display_scores(scores,names=""):
    for name,score in zip(names,scores):
        if names:
            print("Model: ", name)
        print("Scores: ", score)
        print("Mean: ", score.mean())
        print("Std var: ", score.std())

#Fetch data
fetch_housing_data()
housing =  load_housing_data()

#Split test and train data set (additional feature - assure train and test set to be representative for this feature)
housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5, 5.0, inplace = True)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#Check if train and test sets are representative according to new feature 
print(housing["income_cat"].value_counts()/len(housing))
print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))
print(strat_train_set["income_cat"].value_counts()/len(strat_train_set))

#Drop added feature
for set_ in (strat_train_set,strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    
#Split features and labels
housing = strat_train_set.drop("median_house_value",axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
housing_num = housing.drop("ocean_proximity",axis=1) #Numerical features
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

#Data pipelines
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
        ])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CustomLabelBinarizer(sparse_output=False))
        ])
full_pipeline = FeatureUnion(transformer_list = [
        ("num_pipeline",num_pipeline),
        ("cat_pipeline",cat_pipeline)
        ])
housing_prepared = full_pipeline.fit_transform(housing)

#Linear, tree and random forest regression models
lin_reg = LinearRegression()
tree_reg = DecisionTreeRegressor()
forest_reg = RandomForestRegressor()
svm = SVR()

#Cross-validation
results = []
for model in (tree_reg, lin_reg, forest_reg, svm):
    model.fit(housing_prepared,housing_labels)
    scores = cross_val_score(model, housing_prepared, housing_labels, 
                             scoring = "neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    results.append(rmse_scores)
    
display_scores(results,("Tree regressor","Linear regression","Random forest regressor", "SVM"))


#Hyperparameters tuning for Random Forest Regressor (best results obtained)
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2, 3, 4]},
        ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)
grid_search.best_estimator_
features_importance = grid_search.best_estimator_.feature_importances_

#Check performance on test set
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test) #Without fitting!!!
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(final_predictions,y_test)
final_rmse = np.sqrt(final_mse)
print("Final root mean squared error (test set): ", final_rmse)
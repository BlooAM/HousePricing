#!/usr/bin/env python
# coding: utf-8

# # California housing prices prediction

# ## Let's get the data
# At the very beginning, let's write a function to extract data from the tar file named `housing.tgz`

# In[1]:


import os
import tarfile
HOUSING_PATH = os.path.join("data sets","housing")
def fetch_housing_data(housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Then let's define the function for loading extracted csv file to `pandas.DataFrame` object

# In[2]:


import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# ## First glance at the data
# Let's see what the general structure of the data we will be working with looks like

# In[3]:


fetch_housing_data()
housing = load_housing_data()
housing.head()


# We have ten attributes, one of which is categorical and the other numerical.\
# Let's see a summary of all the features

# In[4]:


housing.info()


# Fortunately, there is only one column with missing items (`total_bedrooms`). We will deal with this problem in some time.\
# For categorical attributes, the cardinal number (i.e. how many different classes are possible) is extremely important

# In[5]:


housing['ocean_proximity'].value_counts()


# Five classes is a relatively small number - in the case of one-hot encoding we get only five new features.\
# For numeric data there is a number of statistics that provide some information. The most frequently used are presented below

# In[6]:


housing.describe()


# Their graphic representation is well illustrated by histograms

# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()


# Further exploration of data will be limited to a training set, so as not to suggest too much detail - after all, our model should well generalize the acquired knowledge.
# 
# ## Data preperation
# Create test set for further model validation

# In[8]:


import numpy as np
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


# Note that we are conducting a simple random data separation. This is usually an acceptable approach, but it is good that the test set is as representative as possible for all data (the whole sample). Of course we cannot fully guarantee this, but with some expertise we can force the distribution of the most important features to be similar - it is called stratified sampling.\
# We must therefore consider which feature will play a key role in predictive housing prices - without much doubt we can assume that the average median income is a good candidate. Let us therefore create a new categorical feature corresponding to the aforementioned one

# In[9]:


housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']<5, 5.0, inplace=True)
housing['income_cat'].hist()


# Perform stratified sampling using `StratifiedShuffleSplit` class

# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# The comments made earlier on stratified sampling are well illustrated in the following summary

# In[11]:


def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    "All_data": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Error - random (%)"] = 100 * compare_props["Random"] / compare_props["All_data"] - 100
compare_props["Error - stratified (%)"] = 100 * compare_props["Stratified"] / compare_props["All_data"] - 100
compare_props


# Let us now delete the attribute created only for the purpose of selecting a test set

# In[12]:


for set_ in (strat_test_set, strat_train_set):
    set_.drop('income_cat', axis=1, inplace=True)


# It is advisable to make a copy of the original test set for experimental purposes - especially for feature engineering.

# In[105]:


housing = strat_train_set.copy()


# Let us visualize the data

# In[106]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)


# This graph can be accompanied by information on population density and the median of prices of flats in a given area can be marked with a colour

# In[14]:


housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100, label='Population',
            c=housing['median_house_value'], figsize=(10,7), cmap=plt.get_cmap('jet'), colorbar=True)


# We can conclude that both geographical location and population have a significant impact on the price of a home.\
# The Pearson correlation coefficient additionally indicates how strong is the linear relationship between numerical features

# In[15]:


corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# Its visualization for the most correlated attributes is shown by the scattering graphs below

# In[16]:


from pandas.plotting import scatter_matrix
attributes = ['median_house_value','median_income','total_rooms','housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8))


# As expected, there is a very strong link between the median income and the median house value

# In[17]:


housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)


# The graph shows clear horizontal lines. In addition, there is a clear upper limit for the median value of house prices. It can be deduced from this that this value has an upper limit and each higher median is 'cut' to it. This may cause some obstacles in the learning phase of the model 
# 
# ##  Feature engineering
# 
# This stage is one of the most important in the process of creating a learning system. A good selection of features in relation to a limited amount of data often determines the success of a given project.\
# In order to limit the number of attributes they can be combined - for example, we can create a `Rooms_per_family` feature representing the number of rooms per family in a given district. The same applies to the population `Population_per_family`. Additionally, one could think of the number of bedrooms per room (`Bedrooms_per_room`)

# In[18]:


housing['Rooms_per_family'] = housing['total_rooms']/housing['households']
housing['Bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['Population_per_family'] = housing['population']/housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)


# We have reduced the number of features from 4 to 3, and additionally the newly created two attributes are strongly correlated with the median house value (`Rooms_per_family` , `Bedrooms_per_room`)\
# Data preparation

# In[19]:


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()


# Impute missing values with median

# In[20]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)


# Impute calculated values into train set

# In[21]:


X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)


# Encode categorical attributes

# In[22]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(encoder.classes_)


# Similar features are distant from each other - try one hot encoder

# In[23]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(np.array(housing_cat).reshape(-1,1)) #Direct OneHotEncoder on categorical data
housing_cat_1hot


# Class for previou feature engineering

# In[24]:


from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, houshold_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self #Dummy function
    def transform(self, X, y=None):
        Rooms_per_family = X[:, rooms_ix] / X[:, houshold_ix]
        Population_per_family = X[:, population_ix] / X[:, houshold_ix]
        if self.add_bedrooms_per_room:
            Bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, Rooms_per_family, Population_per_family, Bedrooms_per_room]
        else:
            return np.c_[X, Rooms_per_family, Population_per_family]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


# Create pipeline for imputing, feature engineering and scaling for numerical attributes

# In[25]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


# Data frame handlig for automation

# In[26]:


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


# Create pipelines supporting numerical and categorical features

# In[27]:


num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])


# Pipelines union

# In[28]:


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion([
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# Model learning - linear regression for first shot

# In[29]:


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


# Check performence on training set

# In[30]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions: ', lin_reg.predict(some_data_prepared))
print('Labels: ', list(some_labels))


# RMSE metric for this model

# In[31]:


from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


# Poor performence - try another model

# In[32]:


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)


# Check performance

# In[33]:


housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# Perform cross-validation test

# In[34]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)


# Check results

# In[35]:


def display_scores(scores):
    print('Results: ', scores)
    print('Mean: ', scores.mean())
    print('Std: ', scores.std())

display_scores(tree_rmse_scores)


# Ensamble model - random forest regressor

# In[36]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10)
forest_reg.fit(housing_prepared, housing_labels)
forest_pred = forest_reg.predict(housing_prepared)
forest_rmse = np.sqrt(mean_squared_error(housing_labels,forest_pred))
forest_rmse


# Next check cross-validation

# In[37]:


scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                                  scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)


# Hyperparameters tuning via grid search

# In[38]:


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)


# Print results

# In[39]:


print('Best parameters: ', grid_search.best_params_)
print('Best model: ',grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)


# Feature importance

# In[40]:


feature_importance = grid_search.best_estimator_.feature_importances_
feature_importance


# Link values to coresponding attributes names

# In[41]:


extra_attribs = ['Rooms_per_family', 'Population_per_family', 'Bedroom_per_rooms']
cat_one_hot_attribs = list(encoder.categories_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importance, attributes), reverse=True)


# Test set

# In[42]:


final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse


# ## Support Vector Machine for regression
# Let's try to use an algorithm from different class now - Support Vector Machines. \
# First, let's try to use the default hyperparameter values

# In[43]:


from sklearn.svm import SVR

model = SVR(gamma='auto') #this value will be the default from future versions
svr_scores = cross_val_score(model, housing_prepared, housing_labels, 
                scoring='neg_mean_squared_error', cv=10)
svr_rmse_scores = np.sqrt(-svr_scores)
display_scores(svr_rmse_scores)


# Recall the results for the Random Forest model

# In[44]:


display_scores(forest_rmse_scores)


# Let's try to improve the results by to try out different combinations of hyperparameters: `kernel`, `C` and `gamma`

# In[45]:


param_grid_svr = [
    {'kernel': ['linear'], 'gamma': ['scale', 'auto'], 'C': [0.1, 0.5, 1.0, 2.0, 5.0]},
    {'kernel': ['rbf'], 'gamma': ['scale', 'auto'], 'C': [0.1, 0.5, 1.0, 2.0, 5.0]}
]
grid_search_svr = GridSearchCV(model, param_grid_svr, cv=5, scoring='neg_mean_squared_error')
grid_search_svr.fit(housing_prepared, housing_labels)


# We've searched quite a wide range of parameters. Let's see which model is the best and what results are obtained thanks to it

# In[46]:


print('Best parameters: ', grid_search_svr.best_params_)
print('Best model: ',grid_search_svr.best_estimator_)
cvres = grid_search_svr.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)


# It seems that in this case the linear kernel is doing better. Note that the value of the hyperparameter `C` is the highest value possible - let's test the higher value, leaving the other two hyperparameters unchanged

# In[47]:


param_grid_C = [
    {'C': [5.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]}
]
grid_search_C = GridSearchCV(SVR(kernel='linear', gamma='scale'), param_grid_C, cv=5, scoring='neg_mean_squared_error')
grid_search_C.fit(housing_prepared, housing_labels)


# The results obtained in this way should be even higher - see for yourself

# In[48]:


print('Best parameters: ', grid_search_C.best_params_)
print('Best model: ',grid_search_C.best_estimator_)
cvres = grid_search_C.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)


# The error with respect to the default values has been reduced by almost half! This is a really great result, but the Random Forest model performs even better when the hyperparameter values are adjusted

# ## Tuning via randomized search
# Until now, hyperparameter tuning has consisted in searching the grid for their discreet values - this approach has the advantage that every node of the grid will be examined. If the number of values/attributes increases, the grid size increases powerfully and this approach becomes impractical. Instead, we can use a random search of the grid of parameters

# In[49]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

params = {'kernel': ['linear', 'rbf'], 'C': reciprocal(20,200000)}
random_grid_search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', verbose=2, random_state=42)
random_grid_search.fit(housing_prepared, housing_labels)


# This process took almost 15 minutes, but it was worth it

# In[50]:


print('Best parameters: ', random_grid_search.best_params_)
print('Best model: ',random_grid_search.best_estimator_)
cvres = random_grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)


# Thanks to Randomized Search we got significantly closer to the results of the Random Forest model. It should be noted that this time the Gauss kernel (`rbf`) turned out to be a better choice, which was not noticeable in the case of searching the grid of parameters.\
# The probability distribution (reversed) used for the `C` parameter is illustrated below

# In[52]:


reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reverse distribution")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of reverse distribution")
plt.hist(np.log(samples), bins=50)
plt.show()


# This particular distribution was chosen because it is suitable for estimating hyperparameters for which we do not know the scale (see right figure).\
# ## Automation of feature selection process
# 
# Once we have determined that the Random Forest model is the most appropriate, it would be good to work on the features - we will try to create a function that selects the attributes with the greatest influence (i.e. the best predictors)

# In[67]:


feature_importance = feature_importance[:-1]
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]


# The `TopFeatureSelector` class selects the `k` best features (from the list of `feature_importances`)

# In[69]:


k = 5
top_k_feature_indices = indices_of_top_k(feature_importance, k)
np.array(attributes)[top_k_feature_indices]


# Let us see that the selected features are in fact the most important ones

# In[72]:


sorted(zip(feature_importance, attributes), reverse=True)[:k]


# It is now enough to extend the previously prepared pipeline with a new functionality.

# In[76]:


full_pipeline_feature_selection = Pipeline([
    ('full_pipeline', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importance, k))
])
housing_prepared_top_k_features = full_pipeline_feature_selection.fit_transform(housing)


# It remains to be verified the correctness of the operation of such a pipeline

# In[79]:


np.all(housing_prepared_top_k_features[0:3] == housing_prepared[0:3, top_k_feature_indices])


# ## The final
# 
# We have a full stream that prepares data and selects the most important features. We have also tested several models and tuned the corresponding hyperparameters. It's time to put it all together in one pipeline

# In[84]:


prepare_select_predict_pipeline = Pipeline([
    ('data_preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importance, k)),
    ('prediction', SVR(**random_grid_search.best_params_))
])

prepare_select_predict_pipeline.fit(housing, housing_labels)


# In[87]:


some_data = housing[1:4]
some_labels = housing_labels[1:4]
print("Predictions:\t", prepare_select_predict_pipeline.predict(some_data))
print("Lables:\t\t", list(some_labels))


# The pipeline works, the forecasts are good, and remember that using the model of the Random Forest would be even better!\
# We can now test several strategies for dealing with missing observations and experiment with an optimal number of important features selected to teach the model

# In[100]:


grid_param = [
     {'feature_selection__k': list(range(1, len(feature_importance) + 1)),
     'data_preparation__num_pipeline__imputer__strategy': ['mean', 'median', 'most_frequent'],
     'prediction__gamma': ['auto', 'scale']}
]
grid_search_prep = GridSearchCV(prepare_select_predict_pipeline, grid_param, cv=5,
                               scoring='neg_mean_squared_error', error_score=np.nan)
grid_search_prep.fit(housing, housing_labels)


# Let's read the best settings and scores

# In[104]:


print(grid_search_prep.best_params_)
print(np.sqrt(-grid_search_prep.best_score_))


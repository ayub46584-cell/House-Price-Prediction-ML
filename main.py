import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# 1. Load the data
housing = pd.read_csv("housing.csv")

# 2. Create a stratified test set and train set based on income category using the reference of median income
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# Work on a copy of training data: Because we don't want our training data (strat_train_set) to be affected by any changes we make during data preprocessing
housing = strat_train_set.copy()

# 3. Separate features(predictors) and labels
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)

# 4. Separate numerical and categorical column names
num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()  # .columns() All column names except "ocean_proximity" and .tolist() Converts column names into a Python list
cat_attribs = ["ocean_proximity"] # Returns a python list of a column named "ocean_proximity"

# Note: Why we used ".columns.tolist()"? Because "ColumnTransformer" needs column NAMES in the form of a python list, not column VALUES


# 5. Pipelines
# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    #(name: You can provide any name, transformer: Which transformer you want to use?, columns: Name of the columns on which you want to apply the transformer ?)
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
    #(name, transformer, columns)
])

# 6. Fit and Transform the data
housing_prepared = full_pipeline.fit_transform(housing)



# 7. Train the model on different ML algorithms

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

# Random Forest Regressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)

# Generate predictions (labels) on the training data to evaluate how well the models fit it (useful for detecting overfitting)
lin_preds = lin_reg.predict(housing_prepared)
tree_preds = tree_reg.predict(housing_prepared)
forest_preds = forest_reg.predict(housing_prepared)

# Calculate RMSE for each model
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
tree_rmse = root_mean_squared_error(housing_labels, tree_preds)
forest_rmse = root_mean_squared_error(housing_labels, forest_preds)

print("Linear Regression RMSE: ", lin_rmse)
print("Decision Tree Regressor RMSE: ", tree_rmse) # Output is: 0.0 because Decision Tree Regressor is overfitting the training data, it is memorizing the training data instead of Learning from it.
print("Random Forest Regressor RMSE: ", forest_rmse)

print("\n")

# Cross validation is used to obtain a reliable estimate of model performance by training and testing the model multiple times on different subsets of data.
# We perform cross-validation to: Avoid overfitting, Choose best model etc. It is a powerful technique to evaluate the performance of a machine learning model and to select the best model among different algorithms. 

# 8. Cross-Validation on all the three ML algorithms

# 1. Evaluate Decision Tree Regressor with cross-validation
tree_rmses = -cross_val_score(
    tree_reg, # The model we want to evaluate
    housing_prepared, # Features
    housing_labels, # Labels
    scoring="neg_root_mean_squared_error", # Use RMSE as the evaluation metric, returned as a negative value to follow sklearn's "higher is better" convention
    cv=10 # Number of folds for cross-validation (10-fold cross-validation means the data is split into 10 parts, and the model is trained and evaluated 10 times, each time using a different part as the test set and the remaining parts as the training set)
)

# WARNING: Scikit-Learn’s scoring uses utility functions (higher is better), so RMSE is returned as negative.
# We use minus (-) to convert it back to positive RMSE.

print("Decision Tree Regressor cross validation RMSEs: ", tree_rmses) 
print("\nCross-Validation Performance (Decision Tree Regressor): ")
print(pd.Series(tree_rmses).describe())


print("\n")


# 2. Evaluate Random Forest Regressor with cross-validation
forest_rmses = -cross_val_score(
    forest_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("Random Forest Regressor cross validation RMSEs: ", forest_rmses)
print("\nCross-Validation Performance (Random Forest Regressor): ")
print(pd.Series(forest_rmses).describe())


print("\n")


# 3. Evaluate Linear Regression with cross_validation
lin_rmses = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)
print("Linear Regression cross validation RMSEs: ", lin_rmses)
print("\nCross-Validation Performance (Linear Regression): ")
print(pd.Series(lin_rmses).describe())

# The mean of Decision Tree Regressor is: 69081.361563
# The mean of Random Forest Regressor is: 49432.126788
# The mean of Linear Regression is: 69204.322755
# Based on Cross-Validation Performance, We choose Random Forest Regressor as the best model among the three algorithms. Because it has the lowest RMSE value.
# Now we will use this best model (Random Forest Regressor) to make predictions using Test Set (unseen data) and evaluate its performance. {TESTING THE MODEL}
# This is how a typical Machine Learning Pipeline look's like!
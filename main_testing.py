import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([

        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline = ColumnTransformer([

        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)

    ])

    return full_pipeline

if not os.path.exists(MODEL_FILE):

    # 1. Load data
    housing = pd.read_csv("housing.csv")

    # 2. Create income category for stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # Work on training data
    housing = strat_train_set.copy()

    # 3. Separate features and labels
    housing_labels = housing["median_house_value"].copy()
    housing = housing.drop("median_house_value", axis=1)

    # 4. Separate column names
    num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model and Pipeline saved!")


    # TESTING ON UNSEEN DATA

    # Load test set
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # It reads the saved object from a file 
    loaded_model = joblib.load("MODEL_FILE")
    loaded_pipeline = joblib.load("PIPELINE_FILE")

    # Transform test data (IMPORTANT: only transform, NOT fit). Because here we are testing on unseen data, otherwise it will learn from the test data and we don't want that, we want to evaluate on completely unseen data.
    X_test_prepared = loaded_pipeline.transform(X_test)

    # Predict
    final_predictions = loaded_model.predict(X_test_prepared)

    # Evaluate
    final_rmse = root_mean_squared_error(y_test, final_predictions)

    print("\nFINAL TEST RMSE (Unseen Data): ", final_rmse)

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": final_predictions
    })

    # Save results to CSV
    comparison.to_csv("predictions_vs_actual.csv", index=False)
    print("Saved successfully to predictions_vs_actual.csv, you can open it to see the full comparision of actual vs predicted values.")

else:

    print("Model already exists. Loading and evaluating...")

    # 1. Load data again
    housing = pd.read_csv("housing.csv")

    # 2. Recreate income category (IMPORTANT for same split)
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # 3. Same stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

    # 4. Load test set
    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    # 5. It reads the saved object from a file
    loaded_model = joblib.load(MODEL_FILE)
    loaded_pipeline = joblib.load(PIPELINE_FILE)

    # 6. Transform test data (IMPORTANT: only transform, NOT fit). Because here we are testing on unseen data, otherwise it will learn from the test data and we don't want that, we want to evaluate on completely unseen data.
    X_test_prepared = loaded_pipeline.transform(X_test)

    # 7. Predict
    final_predictions = loaded_model.predict(X_test_prepared)

    # 8. Evaluate
    final_rmse = root_mean_squared_error(y_test, final_predictions)

    print("\nFINAL TEST RMSE (Unseen Data): ", final_rmse)

    # 9. Create comparison DataFrame
    comparison = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": final_predictions
    })

    # Save results to CSV
    comparison.to_csv("predictions_vs_actual.csv", index=False)
    print("Saved successfully to predictions_vs_actual.csv, you can open it to see the full comparision of actual vs predicted values.")
    
    



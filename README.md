# House Price Prediction using Machine Learning

## Project Overview

This project predicts house prices based on various housing features such as:

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Ocean Proximity
- Population
- House holds
- Median Income

The project demonstrates a complete end-to-end Machine Learning workflow including:
- Data preprocessing
- Pipeline creation
- Model training
- Cross validation
- Performance evaluation

---

##  Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn

---

##  ML Workflow

###  Data Splitting
Used **Stratified Shuffle Split** for splitting:
- Training data
- Testing data

---

###  Data Preparation
- Worked on a copy of training data
- Separated:
  - Features
  - Labels

---

###  Pipelines Created

#### Numerical Pipeline
- SimpleImputer
- StandardScaler

#### Categorical Pipeline
- OneHotEncoder

#### Full Pipeline
Combination of:
- Numerical pipeline
- Categorical pipeline

---

##  Algorithms Used

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

##  Model Evaluation

Used:
- RMSE
- Cross Validation

---

##  Best Model

### Random Forest Regressor

Reason:
- Lowest RMSE value
- Better performance compared to other models

---

##  Future Improvements

- Hyperparameter tuning
- Deployment using Flask/Streamlit
- Cloud deployment

---

## 👨‍💻 Author

Mr Ayyub

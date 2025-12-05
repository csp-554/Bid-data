# Household Power Consumption Forecasting  
### Linear Regression, Random Forest, Gradient Boosting, and H2O AutoML

This project performs hourly energy consumption forecasting using the *Individual Household Electric Power Consumption* dataset. The workflow includes data preprocessing, feature engineering, model training using multiple machine learning algorithms, evaluation using MAE and RMSE metrics, and automatic benchmarking using H2O AutoML.

---

## 1. Project Objectives

1. Load and clean the raw household power dataset.
2. Convert timestamps and resample consumption values to hourly averages.
3. Engineer temporal features required for forecasting.
4. Train and evaluate the following regression models:
   - Linear Regression  
   - Random Forest Regressor  
   - Gradient Boosting Regressor  
5. Run H2O AutoML to automatically identify the best performing model.
6. Generate model comparison line and bar charts.
7. Print a clean comparison summary for analysis.

---

## 2. Dataset Description

**Source**: UCI Machine Learning Repository  
**File Used**: `household_power_consumption.txt`

Columns include:
- Date, Time  
- Global Active Power  
- Additional voltage and sub metering values  

This project focuses on forecasting **Global Active Power**.

---

## 3. Processing Workflow

### 3.1 Load and Clean Data
- The dataset is read using `pandas.read_csv`.
- Missing values marked as `?` are properly handled.
- Date and Time columns are merged into a single timestamp.

### 3.2 Resampling
- Data is indexed by timestamp and resampled to **hourly mean consumption**.
- The target variable is defined as the consumption of the next hour.

### 3.3 Feature Engineering
The following features are added:
- hour of day  
- day of week  
- month  
- current hour consumption  

### 3.4 Train Test Split
A chronological split (no shuffling) ensures true time series evaluation.

---

## 4. Models Implemented

### 4.1 Linear Regression
A baseline model used to measure linear relationships in the dataset.

### 4.2 Random Forest Regressor
A strong nonlinear ensemble model that reduces overfitting by averaging multiple trees.

### 4.3 Gradient Boosting Regressor
A sequential boosting model that typically achieves better accuracy than Random Forest on structured data.

Each model outputs:
- MAE (Mean Absolute Error)  
- RMSE (Root Mean Squared Error)

---

## 5. H2O AutoML

H2O AutoML is used to automatically train multiple advanced algorithms including:
- GLM  
- GBM  
- Distributed Random Forest  
- Stacked Ensembles  

The AutoML leaderboard ranks models based on RMSE.

Only compact and necessary information is printed in this project.

---

## 6. Visualizations

The script generates:
- A **line graph** comparing MAE across models.
- A **bar graph** comparing RMSE across models.

These graphs allow quick identification of the most accurate model.

---

## 7. Output Summary Interpretation

Typical model scores obtained:
- Linear Regression MAE: ~0.40  
- Random Forest MAE: ~0.36  
- Gradient Boosting MAE: ~0.34  

And similarly decreasing RMSE values.

Gradient Boosting consistently performs best among the sklearn models.  
H2O AutoML usually produces an even better stacked ensemble model with the lowest RMSE.

---

## 8. Files Included

- `read_data.py`  
  Contains:
  - Data loading  
  - Cleaning and feature engineering  
  - Model training  
  - Evaluation metrics  
  - Visualization  
  - AutoML execution  

---

## 9. Requirements
python3
pandas
numpy
scikit-learn
matplotlib
h2o

Install missing packages using: pip install pandas numpy scikit-learn matplotlib h2o

---

## 10. Running the Script


This will:
1. Load and preprocess data
2. Train all models
3. Display metrics
4. Show graphs
5. Run AutoML and print the leaderboard

---

## 11. Conclusion

This project demonstrates a full workflow for forecasting household electricity consumption using traditional machine learning models and automated ML systems. Gradient Boosting and H2O Stacked Ensembles provide the strongest performance, highlighting the nonlinear and complex relationships in power usage patterns.






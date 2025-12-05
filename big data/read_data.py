import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import h2o
from h2o.automl import H2OAutoML

# ===============================
# 1 Load and prepare dataset
# ===============================

file_path = "household_power_consumption.txt"

df = pd.read_csv(
    file_path,
    sep=";",
    na_values="?",
    low_memory=False
)

# combine date and time
df['timestamp'] = pd.to_datetime(df['Date'] + " " + df['Time'], dayfirst=True)

# index by timestamp
df = df.set_index('timestamp')

# convert to numeric
df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# hourly average
hourly = df['Global_active_power'].resample('h').mean()

# dataframe
hourly_df = hourly.to_frame(name='consumption')

# features
hourly_df['hour'] = hourly_df.index.hour
hourly_df['day_of_week'] = hourly_df.index.dayofweek
hourly_df['month'] = hourly_df.index.month

# prediction target is next hour
hourly_df['target'] = hourly_df['consumption'].shift(-1)

hourly_df = hourly_df.dropna()

# ===============================
# 2 Prepare training data
# ===============================

features = hourly_df[['hour', 'day_of_week', 'month', 'consumption']]
target = hourly_df['target']

X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, shuffle=False
)

# ===============================
# 3 Linear Regression
# ===============================

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

lr_mae = mean_absolute_error(y_test, lr_preds)
lr_rmse = mean_squared_error(y_test, lr_preds) ** 0.5

# ===============================
# 4 Random Forest
# ===============================

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = mean_squared_error(y_test, rf_preds) ** 0.5

# ===============================
# 5 Gradient Boosting
# ===============================

gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)
gb_preds = gb.predict(X_test)

gb_mae = mean_absolute_error(y_test, gb_preds)
gb_rmse = mean_squared_error(y_test, gb_preds) ** 0.5

# ===============================
# 6 Print Summary
# ===============================

print("\n============================")
print("Model Performance Summary")
print("============================")

models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
mae_scores = [lr_mae, rf_mae, gb_mae]
rmse_scores = [lr_rmse, rf_rmse, gb_rmse]

for i, m in enumerate(models):
    print(f"{m}: MAE = {mae_scores[i]:.4f}, RMSE = {rmse_scores[i]:.4f}")

# ===============================
# 7 Plot MAE and RMSE
# ===============================

print("\nPlotting graphs...")

plt.figure(figsize=(8,5))
plt.bar(models, mae_scores, color=['skyblue','lightgreen','salmon'])
plt.title("MAE Comparison Across Models")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(8,5))
plt.bar(models, rmse_scores, color=['skyblue','lightgreen','salmon'])
plt.title("RMSE Comparison Across Models")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(models, mae_scores, marker='o', linewidth=2)
plt.plot(models, rmse_scores, marker='o', linewidth=2)
plt.title("MAE and RMSE Trend Across Models")
plt.xlabel("Model")
plt.ylabel("Error Value")
plt.legend(["MAE", "RMSE"])
plt.grid(True)
plt.show()

# ===============================
# 8 AutoML with H2O
# ===============================

print("\n============================")
print("Running AutoML (H2O)...")
print("============================")

h2o.init()

hf = h2o.H2OFrame(hourly_df)

x = ['hour', 'day_of_week', 'month', 'consumption']
y = 'target'

aml = H2OAutoML(max_runtime_secs=60, seed=42)
aml.train(x=x, y=y, training_frame=hf)

lb = aml.leaderboard
print("\nTop AutoML Models:")
print(lb[['model_id', 'rmse', 'mae']].head())

print("\nBest AutoML Model:")
print(aml.leader.model_id)

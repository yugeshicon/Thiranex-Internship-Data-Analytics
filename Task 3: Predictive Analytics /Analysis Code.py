# Predictive Analytics Using Historical Data - Dataset: Smartphone Dataset

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load Dataset
df = pd.read_csv('3. Predictive Analytics Dataset.csv')
print("\nFirst 5 Rows:\n",df.head())
print("\nDataset Information:\n",df.info())

# Data Cleaning
print("\nMissing Values Before Cleaning:\n",df.isnull().sum())

numerical_columns = ['Battery','Memory','Primary_Storage','Display_Size','Display_Refresh_Rate','Primary_Camera','Front_Camera']
for col in numerical_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

num_imputer = SimpleImputer(strategy='mean')
df[numerical_columns] = num_imputer.fit_transform(df[numerical_columns])

categorical_columns = ['Brand','Model','OS','Processor','External_Storage','Display_Resolution']
cat_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = cat_imputer.fit_transform(df[categorical_columns])

print("\nMissing Values After Cleaning:\n",df.isnull().sum())

# Feature Engineering
df['Release_Date'] = pd.to_datetime(df['Release_Date'], errors='coerce')
df['Release_Year'] = df['Release_Date'].dt.year
df['Release_Year'] = df['Release_Year'].fillna(df['Release_Year'].median())

label_encoder = LabelEncoder()
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column].astype(str))

# Select Features and Target
X = df[['Brand','OS','Memory','Primary_Storage','Display_Size','Primary_Camera','Front_Camera','Release_Year']]
y = df['Battery']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train Machine Learning Model
model = LinearRegression()
model.fit(X_train, y_train)
print("\n*** Model Training Completed Successfully ***")

predictions = model.predict(X_test)
print("\nSample Predictions:\n",predictions[:10])

# Evaluate Model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n*** Model Evaluation Metrics ***\n")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions)
plt.xlabel('Actual Battery Capacity')
plt.ylabel('Predicted Battery Capacity')
plt.title('Actual vs Predicted Battery Capacity')
plt.show()

# Future Prediction: Smartphone Data
future_phone = pd.DataFrame({'Brand': [1], 'OS': [2], 'Memory': [8], 'Primary_Storage': [128], 'Display_Size': [6.5],'Primary_Camera': [50], 'Front_Camera': [16], 'Release_Year': [2026]})
future_prediction = model.predict(future_phone)
print(f"\nPredicted Future Smartphone Battery Capacity: {future_prediction[0]:.2f} mAh")

# Trend Analysis Visualization
trend_data = df.groupby('Release_Year')['Battery'].mean()
plt.figure(figsize=(12, 6))
plt.plot(trend_data.index, trend_data.values, marker='o')
plt.xlabel('Release Year')
plt.ylabel('Average Battery Capacity')
plt.title('Battery Capacity Trend Over Years')
plt.grid(True)
plt.show()

# Project Completed
print("\n*** Predictive Analytics Project Completed Successfully ***")

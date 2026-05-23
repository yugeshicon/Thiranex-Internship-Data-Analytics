# Data Cleaning & Reporting Automation: Smartphone Dataset

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("4. Cleaning & Automation Dataset.csv")
print("\nOriginal Dataset:\n",df.head())
print("\nDataset Information:\n",df.info())
print("\nDataset Shape:\n",df.shape)
      
# Check and Handle Missing Values
print("\nMissing Values Before Cleaning:\n",df.isnull().sum())

numeric_columns = df.select_dtypes(include=np.number).columns
for col in numeric_columns:
    df[col] = df[col].fillna(df[col].mean())

categorical_columns = df.select_dtypes(include='object').columns
for col in categorical_columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Remove Duplicates & Clean The Data
duplicate_count = df.duplicated().sum()
print("\nNumber of Duplicate Rows:", duplicate_count)
df = df.drop_duplicates()

for col in categorical_columns:
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].str.title()

print("\nMissing Values After Cleaning:\n",df.isnull().sum())

# Summary Statistics & Save Cleaned Dataset
print("\nSummary Statistics:\n",df.describe())

df.to_csv("4. Cleaning & Automation Cleaned.csv", index=False)
print("\nCleaned Dataset Saved Successfully!")

# Automated Report Generation
report = pd.DataFrame({"Column Name": df.columns, "Data Type": df.dtypes.values, "Missing Values": df.isnull().sum().values, "Unique Values": df.nunique().values})
print("\nAutomated Data Report:\n",report)

report.to_csv("4. Cleaning & Automation Report.csv", index=False)
print("\nAutomated Report Saved Successfully!")

# Visualization Part

# 1. Histogram
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    plt.hist(df[col], bins=20)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

# 2. Correlation Heatmap
correlation = df[numeric_columns].corr()
plt.figure(figsize=(10,8))
plt.imshow(correlation, cmap='coolwarm', aspect='auto')
plt.colorbar()
plt.xticks(range(len(numeric_columns)), numeric_columns, rotation=90)
plt.yticks(range(len(numeric_columns)), numeric_columns)
plt.title("Correlation Heatmap")
plt.show()

# Top 10 Records After Cleaning
print("\nTop 10 Cleaned Records:\n",df.head(10))

# Project Completed
print("*** Data Cleaning & Reporting Automation Completed Successfully ***")

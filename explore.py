import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Step 1: Look at the first 5 rows
print(df.head())

# Step 2: Check shape (rows, columns)
print('Shape:', df.shape)

# Step 3: Check for missing values
print('Missing values:')
print(df.isnull().sum())

# Step 4: Basic statistics
print(df.describe())

# Step 5: How many diabetic vs non-diabetic?
print(df['Outcome'].value_counts())

# Step 6: Replace 0s in medical columns (they are actually missing values)
cols_with_zeros = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, df[cols_with_zeros].mean())

# Save cleaned data
df.to_csv('diabetes_clean.csv', index=False)
print('Cleaned data saved!')
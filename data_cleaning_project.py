# data_cleaning_project.py

import pandas as pd
import numpy as np

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

print("Original Data Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())


print("\nMissing Values Before Cleaning:")
print(df.isnull().sum())


df['age'].fillna(df['age'].median(), inplace=True)


df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)


df.drop(columns=['deck'], inplace=True)

df.drop_duplicates(inplace=True)


df['age'] = df['age'].astype(int)


df['fare'] = df['fare'].astype(float)


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

df['embarked'] = df['embarked'].str.upper().str.strip()


q1 = df['fare'].quantile(0.25)
q3 = df['fare'].quantile(0.75)
iqr = q3 - q1
upper_limit = q3 + 1.5 * iqr
df = df[df['fare'] <= upper_limit]


print("\nData Shape After Cleaning:", df.shape)
print("\nCleaned Data Sample:")
print(df.head())


df.to_csv("cleaned_titanic.csv", index=False)
print("\nCleaned dataset saved as 'cleaned_titanic.csv'")

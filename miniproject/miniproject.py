import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
file_path = r'C:\AI\miniproject\cleaned_data.csv'
data = pd.read_csv(file_path)

# Basic statistics
numeric_cols = ['age', 'latitude', 'longitude']
print("Basic Statistics for Numeric Columns:")
print(data[numeric_cols].describe())
print("\nMedian Values:")
print(data[numeric_cols].median())

# Gender distribution
gender_dist = data['gender'].value_counts(normalize=True) * 100
print("\nGender Distribution (%):\n", gender_dist)

# Vehicle type distribution
vehicle_dist = data['vehicle_type'].value_counts(normalize=True) * 100
print("\nTop 10 Vehicle Types (%):\n", vehicle_dist.head(10))

# Provinces with highest accidents
print("\nTop 10 Provinces with Most Accidents:\n", data['province_en'].value_counts().head(10))

# Age distribution histogram
plt.figure(figsize=(10, 6))
age_counts, age_edges = np.histogram(data['age'].dropna(), bins=20)
age_max_idx = np.argmax(age_counts)
age_max_range = (age_edges[age_max_idx], age_edges[age_max_idx + 1])
sns.histplot(data['age'].dropna(), color='skyblue', bins=20)
plt.axvspan(*age_max_range, color='red', alpha=0.3, label=f'Most Accidents: {age_max_range[0]:.1f}-{age_max_range[1]:.1f} years')
plt.title('Age Distribution of Accident Victims', fontsize=16, weight='bold')
plt.xlabel('Age (years)', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.legend(fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Gender distribution bar plot
gender_dist.plot(kind='bar', color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Gender Distribution', fontsize=16, weight='bold')
plt.xlabel('Gender', fontsize=14)
plt.ylabel('Percentage (%)', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Average age for top 5 accident causes
top_causes = data['accident_cause_code'].value_counts().head(5).index
avg_age_per_cause = data[data['accident_cause_code'].isin(top_causes)].groupby('accident_cause_code')['age'].mean()
avg_age_per_cause.sort_values().plot(kind='bar', color='skyblue', edgecolor='black', figsize=(10, 6))
plt.title('Average Age by Top 5 Accident Causes', fontsize=16, weight='bold')
plt.xlabel('Accident Cause Code', fontsize=14)
plt.ylabel('Average Age (years)', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Heatmap: Vehicle type vs Accident causes
filtered_data = data[data['accident_cause_code'].isin(data['accident_cause_code'].value_counts().head(10).index)]
cross_table = pd.crosstab(filtered_data['vehicle_type'], filtered_data['accident_cause_code'])
plt.figure(figsize=(14, 8))
sns.heatmap(cross_table, annot=True, fmt='d', cmap='coolwarm', cbar_kws={'label': 'Frequency'})
plt.title('Vehicle Type vs Accident Causes', fontsize=16, weight='bold')
plt.xlabel('Accident Cause Code', fontsize=14)
plt.ylabel('Vehicle Type', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
plt.yticks(fontsize=12)
plt.show()

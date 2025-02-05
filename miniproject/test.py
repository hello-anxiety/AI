import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# โหลดข้อมูลจากไฟล์ CSV
Data = pd.read_csv(r'C:\AI\miniproject\cleaned_data.csv')

# สถิติของ colums ตัวเลข
numeric_cols = ['age', 'latitude', 'longitude']
print("สถิติพื้นฐานของคอลัมน์ตัวเลข:")
print(Data[numeric_cols].describe())

# คำนวนมัธยฐาน
print("\nค่ามัธยฐาน (Median):")
print(Data[numeric_cols].median())

# การกระจายของเพศ
gender_dist = Data['gender'].value_counts(normalize=True) * 100
print("\nการกระจายเพศ:")
print(gender_dist)


# การกระจายของประเภทยานพาหนะ
vehicle_dist = Data['vehicle_type'].value_counts(normalize=True) * 100
print("\nการกระจายประเภทยานพาหนะ:")
print(vehicle_dist.head(10)) 

# จังหวัดที่มีอุบัติเหตุสูงสุด
print("\nจังหวัดที่มีอุบัติเหตุสูงสุด 10 อันดับแรก:")
print(Data['province_en'].value_counts().head(10))


# สร้างฮิสโตแกรม
plt.figure(figsize=(10, 6))
hist_plot = sns.histplot(Data['age'], color='skyblue', bins=20)

# หาช่วงอายุที่มีการเกิดอุบัติเหตุสูงสุด
counts, edges = np.histogram(Data['age'], bins=20)  # ใช้ dropna() เพื่อจัดการกับค่าว่าง
max_bin_index = np.argmax(counts)  # ดัชนีของ bin ที่มีจำนวนสูงสุด
max_range = (edges[max_bin_index], edges[max_bin_index + 1])  # ช่วงอายุของ bin นั้น

# เพิ่มข้อความระบุช่วงอายุที่เกิดอุบัติเหตุมากที่สุด
plt.axvspan(max_range[0], max_range[1], color='red', alpha=0.3, label=f'Most accidents: {max_range[0]:.1f}-{max_range[1]:.1f} years')

# ตั้งค่าและแสดงกราฟ
plt.title('Age Distribution of Accident Victims', fontsize=16, weight='bold')
plt.xlabel('Age (years)', fontsize=14)
plt.ylabel('Quantity', fontsize=14)
plt.legend(loc='upper right', fontsize=12)
plt.show()

#เพศ vs อุบัติดหตุ
gender_dist.plot(kind='bar', color=['skyblue', 'salmon'], edgecolor="black")
plt.title('Gender Distribution', fontsize=16, weight='bold')
plt.xlabel('Gender', fontsize=12)
plt.ylabel('Percentage', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# คำนวณอายุเฉลี่ยสำหรับสาเหตุยอดนิยม 5 อันดับแรก
top_causes = Data['accident_cause_code'].value_counts().head(5).index
avg_age_per_cause = Data[Data['accident_cause_code'].isin(top_causes)].groupby('accident_cause_code')['age'].mean()

# สร้างกราฟแท่ง
plt.figure(figsize=(10, 6))
avg_age_per_cause.sort_values().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Average Age by Top 5 Accident Causes', fontsize=16, weight='bold')
plt.xlabel('Accident Cause Code', fontsize=14)
plt.ylabel('Average Age (years)', fontsize=14)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Select top accident causes to reduce clutter
top_causes = Data['accident_cause_code'].value_counts().head(10).index
filtered_data = Data[Data['accident_cause_code'].isin(top_causes)]
    
# Create crosstab
cross_table = pd.crosstab(filtered_data['vehicle_type'], filtered_data['accident_cause_code'])
    
# Plot heatmap
plt.figure(figsize=(14, 8))
sns.heatmap(cross_table, annot=True, fmt='d', cmap='coolwarm', cbar_kws={'label': 'Frequency'})
    
# Improve readability
plt.title('Vehicle Type vs Accident Causes', fontsize=16, weight='bold')
plt.xlabel('Accident Cause Code', fontsize=12)
plt.ylabel('Vehicle Type', fontsize=12)
plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels
plt.yticks(fontsize=10)  # Adjust y-axis font size    
plt.show()



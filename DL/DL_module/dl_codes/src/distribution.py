import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the generated dataset
df = pd.read_csv(r'D:\brototype\week27\DL\DL_module\dl_codes\data\generated_data.csv')

# 1. Check for Missing Values 
print("Dataset Information (Initial Check)")
df.info()

# 2. Calculate and print the distribution of the target variable 
prognosis_counts = df['prognosis'].value_counts()
print("\n Class Distribution (Target Variable: 'prognosis')")
print(prognosis_counts)

# 3. Visualize the class distribution 
plt.figure(figsize=(12, 6))
sns.barplot(
    y=prognosis_counts.index, 
    x=prognosis_counts.values,
    hue=prognosis_counts,
    legend=False, 
    orient='h', 
    palette="viridis"
)
plt.title('Distribution of Disease Prognosis')
plt.xlabel('Number of Samples')
plt.ylabel('Disease')
plt.tight_layout()
plt.show() 

# 4. Check the distribution of a numeric feature 
plt.figure(figsize=(10, 5))
sns.histplot(
    df['symptom_count'], 
    bins=range(int(df['symptom_count'].max()) + 2), # Bins for integer counts
    kde=False, 
    color='skyblue'
)
plt.title('Distribution of Symptom Counts per Sample')
plt.xlabel('Number of Symptoms')
plt.ylabel('Frequency')
plt.xticks(range(int(df['symptom_count'].max()) + 1))
plt.grid(axis='y', alpha=0.5)
plt.tight_layout()
plt.show()
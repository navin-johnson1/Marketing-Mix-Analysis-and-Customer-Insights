#Importing all the required Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

#Configuration
sns.set(style="whitegrid")

#Load and Clean the Dataset
file_path = "marketing_data.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

#Clean Income column
df['Income'] = df['Income'].replace('[\$,]', '', regex=True).astype(float)

#Convert Dt_Customer to datetime
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], errors='coerce')

#Clean Categorical Variables
df['Education'] = df['Education'].str.title().str.strip()
df['Marital_Status'] = df['Marital_Status'].replace({
    'Alone': 'Single', 'Absurd': 'Single', 'Yolo': 'Single', 'Widow': 'Widowed'
})

#Impute Missing Income by Group Median
df['Income'] = df.groupby(['Education', 'Marital_Status'])['Income'].transform(lambda x: x.fillna(x.median()))

#Feature Engineering
df['Age'] = 2025 - df['Year_Birth']
df['Total_Children'] = df['Kidhome'] + df['Teenhome']
spend_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
df['Total_Spending'] = df[spend_cols].sum(axis=1)
df['Total_Purchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].sum(axis=1)

#Outlier Detection & Treatment
income_cap = df['Income'].quantile(0.99)
df['Income'] = np.where(df['Income'] > income_cap, income_cap, df['Income'])

#Encoding
edu_order = ['Basic', '2N Cycle', 'Graduation', 'Master', 'Phd']
df['Education'] = pd.Categorical(df['Education'], categories=edu_order, ordered=True).codes
df = pd.get_dummies(df, columns=['Country'], drop_first=True)

#Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

#Hypothesis Testing
# Age vs Store Purchases
print("Creating Age bins for boxplot...")
df['Age_Bin'] = pd.qcut(df['Age'], 4)
sns.boxplot(x='Age_Bin', y='NumStorePurchases', data=df)
plt.title('Age vs Store Purchases')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Children vs Web Purchases
sns.boxplot(x='Total_Children', y='NumWebPurchases', data=df)
plt.title('Children at Home vs Web Purchases')
plt.tight_layout()
plt.show()

#Correlation between Store and Web Purchases:
print("Correlation between Store and Web Purchases:")
print(df[['NumStorePurchases', 'NumWebPurchases']].corr())

#US vs Non-US Purchase Comparison
us_col = 'Country_US'
if us_col in df.columns:
    df['Is_US'] = df[us_col]
    us_total = df[df['Is_US'] == 1]['Total_Purchases']
    non_us_total = df[df['Is_US'] == 0]['Total_Purchases']
    stat, p = ttest_ind(us_total, non_us_total)
    print(f"T-Test for US vs Non-US Total Purchases: t-stat = {stat:.2f}, p-value = {p:.4f}")

#Visual Insights
#Product Spend Totals
product_totals = df[spend_cols].sum().sort_values(ascending=False)
product_totals.plot(kind='bar', title='Product Spend Totals')
plt.ylabel('Total Spending')
plt.tight_layout()
plt.show()

#Age vs Campaign Response
sns.boxplot(x='Response', y='Age', data=df)
plt.title('Campaign Response vs Age')
plt.tight_layout()
plt.show()

#Country with Most Acceptances
if 'Response' in df.columns:
    df_accepts = df[df['Response'] == 1]
    country_cols = [col for col in df.columns if col.startswith('Country_')]
    df_accepts[country_cols].sum().sort_values(ascending=False).plot(kind='bar', title='Campaign Acceptances by Country')
    plt.ylabel('Number of Acceptances')
    plt.tight_layout()
    plt.show()

#Children vs Total Spending
sns.boxplot(x='Total_Children', y='Total_Spending', data=df)
plt.title('Children at Home vs Total Spending')
plt.tight_layout()
plt.show()

#Complaints by Education
if 'Complain' in df.columns:
    df[df['Complain'] == 1]['Education'].value_counts().sort_index().plot(kind='bar', title='Education of Complaining Customers')
    plt.ylabel('Count')
    plt.xlabel('Education Code')
    plt.tight_layout()
    plt.show()
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
os.system('cls' if os.name == 'nt' else 'clear'); plt.close('all');

# %%main variable definitions
#read data from training file and droppping unnecessary data

Q = 1.5
min_corr = 0.5
max_cut = 0.3

data = pd.read_csv('train.csv')
data.replace({'1stFlrSF': np.nan}, 0, inplace=True)
data.replace({'2ndFlrSF': np.nan}, 0, inplace=True)
data.replace({'TotalBsmtSF': np.nan}, 0, inplace=True)
data.replace({'GarageArea': np.nan}, 0, inplace=True)
data['TotalArea'] = data['TotalBsmtSF'] + data['2ndFlrSF'] + data['1stFlrSF'] + data['GarageArea']

data = data.drop('Id', axis=1).drop('MSSubClass', axis=1).drop('1stFlrSF', axis=1).drop('2ndFlrSF', axis=1).drop('TotalBsmtSF', axis=1).drop('GrLivArea', axis=1).drop('GarageArea', axis=1).drop('BsmtFinSF2', axis=1)

#seperate the data into numerical and nominal
 
num_col = data.select_dtypes(include='number');
nom_col = data.select_dtypes(include='object');

num_col = num_col.fillna(0) #replace not-a-numbers with zeros in numerical data

discrete_num_col = []
continuous_num_col = []

for col in num_col.columns:
    unique_vals = num_col[col].nunique()
    if unique_vals <= 15:
        discrete_num_col.append(col)
    else:
        continuous_num_col.append(col)

continuous_num_col = num_col[continuous_num_col].copy()
discrete_num_col = num_col[discrete_num_col].copy()

# %% the continiuous linear case
#use IQR filtering to remove outliers with a maximum filtering range of 30%

m = []; P = np.arange(0, max_cut/2.0, 0.01);

#choose the range (1 - p) with the max linear correlation in normal axis

for p in P:
    Q1 = continuous_num_col.quantile(p); Q3 = continuous_num_col.quantile(1 - p); IQR = Q3 - Q1;
    filtered = continuous_num_col[~((continuous_num_col < (Q1 - Q * IQR)) |  (continuous_num_col > (Q3 + Q * IQR))).any(axis=1)]
    m.append(abs(filtered.corr()['SalePrice']).drop('SalePrice').mean())

#finding the best range (p)

m = np.array(m)
p = P[np.argmax(m[~np.isnan(m)])];
Q1 = continuous_num_col.quantile(p); Q3 = continuous_num_col.quantile(1 - p); IQR = Q3 - Q1
continuous_num_col_linear = continuous_num_col[~((continuous_num_col < (Q1 - Q * IQR)) | (continuous_num_col > (Q3 + Q * IQR))).any(axis=1)]   
    
#checking correlation of all "inputs" with the "output" which is the sale price then removing self correlation

crlt_linear = abs(continuous_num_col_linear.corr()['SalePrice']).drop('SalePrice').sort_values(ascending=False)

#choosing the data which "moderatly" correlates (correlation higher than 0.6) with the output

crlt_linear = crlt_linear[(crlt_linear >= min_corr)]

#determining subplot size 

if (len(crlt_linear) % 2) == 0:
    fig, axs = plt.subplots(int(len(crlt_linear)//2), 2, figsize=(10, 8))
else:
    fig, axs = plt.subplots(int(len(crlt_linear) + 1)//2, 2, figsize=(10, 8))

axs = axs.flatten()

#plotting all the moderatly correlating inputs with the output

for n in range(len(crlt_linear)):
    axs[n].plot(continuous_num_col_linear[crlt_linear.index[n]], continuous_num_col_linear['SalePrice'], linestyle='', marker='.')
    axs[n].grid(True); axs[n].set_xlabel(crlt_linear.index[n]); axs[n].set_ylabel("Sale Price");

plt.tight_layout()
plt.show()

# %% the continuous power case
#use IQR filtering to remove outliers with a maximum filtering range of 30%

m = []; P = np.arange(0,  max_cut/2.0, 0.01);

#choose the range (1 - p) with the max power correlation in loglog axis

continuous_num_col_log = np.log(continuous_num_col)

#turn all inf into nan values

continuous_num_col_log.replace([np.inf, -np.inf], np.nan, inplace=True)

for p in P:
    Q1 = continuous_num_col_log.quantile(p); Q3 = continuous_num_col_log.quantile(1 - p); IQR = Q3 - Q1;
    filtered = continuous_num_col_log[~((continuous_num_col_log < (Q1 - Q * IQR)) |  (continuous_num_col_log > (Q3 + Q * IQR))).any(axis=1)]
    m.append(abs(filtered.corr()['SalePrice']).drop('SalePrice').mean())

#finding the best range (p)

m = np.array(m)
p = P[np.argmax(m[~np.isnan(m)])];
Q1 = continuous_num_col_log.quantile(p); Q3 = continuous_num_col_log.quantile(1 - p); IQR = Q3 - Q1
continuous_num_col_power = continuous_num_col_log[~((continuous_num_col_log < (Q1 - Q * IQR)) | (continuous_num_col_log > (Q3 + Q * IQR))).any(axis=1)]   

#checking correlation of all "inputs" with the "output" which is the sale price then removing self correlation

crlt_power = abs(continuous_num_col_power.corr()['SalePrice']).drop('SalePrice').sort_values(ascending=False)

#choosing the data which "moderatly" correlates (correlation higher than 0.6) with the output

crlt_power = crlt_power[(crlt_power >= min_corr)]

#determining subplot size 

if (len(crlt_power) % 2) == 0:
    fig, axs = plt.subplots(int(len(crlt_power)//2), 2, figsize=(10, 8))
else:
    fig, axs = plt.subplots(int(len(crlt_power) + 1)//2, 2, figsize=(10, 8))

axs = axs.flatten()

#plotting all the moderatly correlating inputs with the output

for n in range(len(crlt_power)):
    axs[n].plot(continuous_num_col_power[crlt_power.index[n]], continuous_num_col_power['SalePrice'], linestyle='', marker='.')
    axs[n].grid(True); axs[n].set_xlabel(f"log({crlt_power.index[n]})"); axs[n].set_ylabel("log(Sale Price)")


plt.tight_layout()
plt.show()

# %% the continuous exponential case
#use IQR filtering to remove outliers with a maximum filtering range of 30%

m = []; P = np.arange(0,  max_cut/2.0, 0.01);

#choose the range (1 - p) with the max exponential correlation in normal axis

continuous_num_col_exp = continuous_num_col.copy();
continuous_num_col_exp['SalePrice'] = np.log(continuous_num_col['SalePrice'])

for p in P:
    Q1 = continuous_num_col_exp.quantile(p); Q3 = continuous_num_col_exp.quantile(1 - p); IQR = Q3 - Q1;
    filtered = continuous_num_col_exp[~((continuous_num_col_exp < (Q1 - Q * IQR)) |  (continuous_num_col_exp > (Q3 + Q * IQR))).any(axis=1)]
    m.append(abs(filtered.corr()['SalePrice']).drop('SalePrice').mean())

#finding the best range (p)
m = np.array(m)
p = P[np.argmax(m[~np.isnan(m)])];
Q1 = continuous_num_col_exp.quantile(p); Q3 = continuous_num_col_exp.quantile(1 - p); IQR = Q3 - Q1
continuous_num_col_exp = continuous_num_col_exp[~((continuous_num_col_exp < (Q1 - Q * IQR)) | (continuous_num_col_exp > (Q3 + Q * IQR))).any(axis=1)]   
    
#checking correlation of all "inputs" with the "output" which is the sale price then removing self correlation

crlt_exp = abs(continuous_num_col_exp.corr()['SalePrice']).drop('SalePrice').sort_values(ascending=False)

#choosing the data which "moderatly" correlates (correlation higher than 0.6) with the output

crlt_exp = crlt_exp[(crlt_exp >= min_corr)]

#determining subplot size 

if (len(crlt_exp) % 2) == 0:
    fig, axs = plt.subplots(int(len(crlt_exp)//2), 2, figsize=(10, 8))
else:
    fig, axs = plt.subplots(int(len(crlt_exp) + 1)//2, 2, figsize=(10, 8))

axs = axs.flatten()

#plotting all the moderatly correlating inputs with the output

for n in range(len(crlt_exp)):
    axs[n].plot(continuous_num_col_exp[crlt_exp.index[n]], continuous_num_col_exp['SalePrice'], linestyle='', marker='.')
    axs[n].grid(True); axs[n].set_xlabel(crlt_exp.index[n]); axs[n].set_ylabel("log(Sale Price)");

plt.tight_layout()
plt.show()

# %% catagorization by price
catagory = 'SalePrice'
splits = 4
max_val_cat = np.max(continuous_num_col[catagory])

# Define bins and labels
bins = np.arange(0, ((splits + 1.0)/splits) * max_val_cat, max_val_cat / splits)
labels = [f'Group {i+1}' for i in range(len(bins) - 1)]

# Assign each row to a price group
bin_indices = pd.cut(continuous_num_col[catagory], bins=bins, labels=labels, include_lowest=True)
split_groups = [continuous_num_col[bin_indices == label][catagory] for label in labels]

# Initialize list to store (feature, correlation) pairs
group_corr_values = [[] for _ in range(splits)]

for i in range(len(split_groups)):
    group_data = continuous_num_col.loc[split_groups[i].index]
    corr_vec = group_data.corr()[catagory].drop(catagory)
    corr_vec.replace(np.nan, 0, inplace=True)

    # Get top 3 correlations by absolute value
    top_corr = corr_vec.reindex(corr_vec.abs().sort_values(ascending=False).index)[:3]
    group_corr_values[i] = list(zip(top_corr.index, top_corr.values))
    
#creating tbale via a list of dictionaries for each group
summary_rows = []

for i, group in enumerate(group_corr_values):
    for feature, corr in group:
        summary_rows.append({
            catagory + ' Group': labels[i],
            'Feature': feature,
            'Correlation': round(corr, 3)
        })

#convert table to DataFrame
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(by=[catagory + ' Group', 'Correlation'], ascending=[True, False])

fig, ax = plt.subplots(figsize=(6, len(summary_df))) 
ax.axis('off')

table = ax.table(
    cellText=summary_df.values,
    colLabels=summary_df.columns,
    cellLoc='center',
    loc='center'
)

plt.title("Top Correlated Features per Group", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

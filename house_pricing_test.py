import os
import numpy as npy
import pandas as pds
import matplotlib.pyplot as plt
os.system('cls')

# %%
#read data from training file and droppping unnecessary data


data = pds.read_csv('train.csv').drop('Id', axis=1).drop('MSSubClass', axis=1) 

#seperate the data into numerical and nominal
 
num_col = data.select_dtypes(include='number');
nom_col = data.select_dtypes(include='object');

num_col = num_col.fillna(0) #replace not-a-numbers with zeros in numerical data

# %% the linear case
#use IQR filtering to remove outliers with a maximum filtering range of 30%

m = []; P = npy.arange(0, 0.30, 0.01);

#choose the range (1 - p) with the max linear correlation in normal axis

for p in P:
    Q1 = num_col.quantile(p); Q3 = num_col.quantile(1 - p); IQR = Q3 - Q1;
    filtered = num_col[~((num_col < (Q1 - 1.5 * IQR)) |  (num_col > (Q3 + 1.5 * IQR))).any(axis=1)]
    m.append(abs(filtered.corr()['SalePrice']).drop('SalePrice').mean())

#finding the best range (p)

m = npy.array(m)
p = P[npy.argmax(m[~npy.isnan(m)])];
Q1 = num_col.quantile(p); Q3 = num_col.quantile(1 - p); IQR = Q3 - Q1
num_col_linear = num_col[~((num_col < (Q1 - 1.5 * IQR)) | (num_col > (Q3 + 1.5 * IQR))).any(axis=1)]   
    
#checking correlation of all "inputs" with the "output" which is the sale price then removing self correlation

crlt_linear = abs(num_col_linear.corr()['SalePrice']).drop('SalePrice').sort_values(ascending=False)

#choosing the data which "moderatly" correlates (correlation higher than 0.6) with the output

crlt_linear = crlt_linear[(crlt_linear >= 0.6)]

#determining subplot size 

if (len(crlt_linear) % 2) == 0:
    fig, axs = plt.subplots(int(len(crlt_linear)//2), 2, figsize=(10, 8))
else:
    fig, axs = plt.subplots(int(len(crlt_linear) + 1)//2, 2, figsize=(10, 8))

axs = axs.flatten()

#plotting all the moderatly correlating inputs with the output

for n in range(len(crlt_linear)):
    axs[n].plot(num_col_linear[crlt_linear.index[n]], num_col_linear['SalePrice'], linestyle='', marker='.')
    axs[n].grid(True); axs[n].set_xlabel(crlt_linear.index[n]); axs[n].set_ylabel("Sale Price");

plt.tight_layout()
plt.show()

# %% the power case
#use IQR filtering to remove outliers with a maximum filtering range of 30%

m = []; P = npy.arange(0, 0.3, 0.01);

#choose the range (1 - p) with the max power correlation in loglog axis

num_col_log = npy.log(num_col)

#turn all inf into nan values

num_col_log.replace([npy.inf, -npy.inf], npy.nan, inplace=True)

for p in P:
    Q1 = num_col_log.quantile(p); Q3 = num_col_log.quantile(1 - p); IQR = Q3 - Q1;
    filtered = num_col_log[~((num_col_log < (Q1 - 1.5 * IQR)) |  (num_col_log > (Q3 + 1.5 * IQR))).any(axis=1)]
    m.append(abs(filtered.corr()['SalePrice']).drop('SalePrice').mean())

#finding the best range (p)

m = npy.array(m)
p = P[npy.argmax(m[~npy.isnan(m)])];
Q1 = num_col_log.quantile(p); Q3 = num_col_log.quantile(1 - p); IQR = Q3 - Q1
num_col_power = num_col_log[~((num_col_log < (Q1 - 1.5 * IQR)) | (num_col_log > (Q3 + 1.5 * IQR))).any(axis=1)]   

#checking correlation of all "inputs" with the "output" which is the sale price then removing self correlation

crlt_power = abs(num_col_power.corr()['SalePrice']).drop('SalePrice').sort_values(ascending=False)

#choosing the data which "moderatly" correlates (correlation higher than 0.6) with the output

crlt_power = crlt_power[(crlt_power >= 0.6)]

#determining subplot size 

if (len(crlt_power) % 2) == 0:
    fig, axs = plt.subplots(int(len(crlt_power)//2), 2, figsize=(10, 8))
else:
    fig, axs = plt.subplots(int(len(crlt_power) + 1)//2, 2, figsize=(10, 8))

axs = axs.flatten()

#plotting all the moderatly correlating inputs with the output

for n in range(len(crlt_power)):
    axs[n].plot(num_col_power[crlt_power.index[n]], num_col_power['SalePrice'], linestyle='', marker='.')
    axs[n].grid(True); axs[n].set_xlabel(f"log({crlt_power.index[n]})"); axs[n].set_ylabel("log(Sale Price)")


plt.tight_layout()
plt.show()

# %% the exponential case
#use IQR filtering to remove outliers with a maximum filtering range of 30%

m = []; P = npy.arange(0, 0.30, 0.01);

#choose the range (1 - p) with the max exponential correlation in normal axis

num_col_exp = num_col;
num_col_exp['SalePrice'] = npy.log(num_col_exp['SalePrice'])

for p in P:
    Q1 = num_col_exp.quantile(p); Q3 = num_col_exp.quantile(1 - p); IQR = Q3 - Q1;
    filtered = num_col_exp[~((num_col_exp < (Q1 - 1.5 * IQR)) |  (num_col_exp > (Q3 + 1.5 * IQR))).any(axis=1)]
    m.append(abs(filtered.corr()['SalePrice']).drop('SalePrice').mean())

#finding the best range (p)

m = npy.array(m)
p = P[npy.argmax(m[~npy.isnan(m)])];
Q1 = num_col_exp.quantile(p); Q3 = num_col_exp.quantile(1 - p); IQR = Q3 - Q1
num_col_exp = num_col_exp[~((num_col_exp < (Q1 - 1.5 * IQR)) | (num_col_exp > (Q3 + 1.5 * IQR))).any(axis=1)]   
    
#checking correlation of all "inputs" with the "output" which is the sale price then removing self correlation

crlt_exp = abs(num_col_exp.corr()['SalePrice']).drop('SalePrice').sort_values(ascending=False)

#choosing the data which "moderatly" correlates (correlation higher than 0.6) with the output

crlt_exp = crlt_exp[(crlt_exp >= 0.6)]

#determining subplot size 

if (len(crlt_exp) % 2) == 0:
    fig, axs = plt.subplots(int(len(crlt_exp)//2), 2, figsize=(10, 8))
else:
    fig, axs = plt.subplots(int(len(crlt_exp) + 1)//2, 2, figsize=(10, 8))

axs = axs.flatten()

#plotting all the moderatly correlating inputs with the output

for n in range(len(crlt_exp)):
    axs[n].plot(num_col_exp[crlt_exp.index[n]], num_col_exp['SalePrice'], linestyle='', marker='.')
    axs[n].grid(True); axs[n].set_xlabel(crlt_exp.index[n]); axs[n].set_ylabel("log(Sale Price)");

plt.tight_layout()
plt.show()

# %% dealing with nominal values
#initialize variables getting the unique values for nominal values (to be continued)


vals_nom = []

for col in nom_col.columns:
    vals_nom.append(nom_col[col].unique())
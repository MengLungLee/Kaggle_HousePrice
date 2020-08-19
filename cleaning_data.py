#%%
import pandas as pd
import numpy as np

raw_train_data = pd.read_csv('train.csv')
raw_train_data.drop('Id', inplace = True, axis = 1)

raw_test_data = pd.read_csv('test.csv')
raw_test_data.drop('Id', inplace = True, axis = 1)

ntrain = raw_train_data.shape[0] # record the size of trainset

raw_all_data = pd.concat((raw_train_data,raw_test_data)).reset_index(drop = True)
#%%
# check missing data and drop out > 15% features
total = raw_all_data.isnull().sum().sort_values()
percentage = (raw_all_data.isnull().sum() / raw_all_data.isnull().count()).sort_values(ascending = False)

df = raw_all_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)

# %%
# filling out missing data left and remove some features from domain knowledge.
df = df.drop(['GarageType', 'GarageFinish'], axis = 1)
for val in df[['GarageCond', 'GarageQual']]: 
     df[val] = df[val].fillna('None')
for val in df[['GarageYrBlt', 'GarageArea', 'GarageCars']]:
    df[val] = df[val].fillna(0)

df['GarageYrBlt'] = df['GarageYrBlt'].apply(lambda x: (2020 - x) if x > 0 else 0)

df = df.drop(['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis = 1)
for val in df[['BsmtQual', 'BsmtCond']]:
    df[val] = df[val].fillna('None')

df['MasVnrType'] = df['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

categorical = ['MSZoning', 'Electrical', 'Utilities', 'Exterior1st', 'Exterior2nd','KitchenQual','Functional','SaleType']
for col in categorical:
    df[col] = df[col].fillna(df[col].mode()[0])

numerical = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
for num in numerical:
    df[num] = df[num].fillna(0)

# %%
df_out_train = df.iloc[:ntrain, :]
df_out_train.to_csv('train_cleaned.csv', index = False)

df_out_test = df.iloc[ntrain:,:]
df_out_test.to_csv('test_cleaned.csv', index = False)
# %%

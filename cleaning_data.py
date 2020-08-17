#%%
import pandas as pd
import numpy as np

raw_data = pd.read_csv('train.csv')
raw_data.drop('Id', inplace = True, axis = 1)

raw_test_data = pd.read_csv('test.csv')
raw_test_data.drop('Id', inplace = True, axis = 1)
#%%
# check missing data and drop out > 15% features
total = raw_data.isnull().sum().sort_values()
percentage = (raw_data.isnull().sum() / raw_data.isnull().count()).sort_values(ascending = False)

df = raw_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)
df_test = raw_test_data.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage'], axis=1)
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

df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

#%%
# for test part
df_test = df_test.drop(['GarageType', 'GarageFinish'], axis = 1)
for val in df_test[['GarageCond', 'GarageQual']]: 
     df_test[val] = df_test[val].fillna('None')
for val in df_test[['GarageYrBlt', 'GarageArea', 'GarageCars']]:
    df_test[val] = df_test[val].fillna(0)

df_test['GarageYrBlt'] = df_test['GarageYrBlt'].apply(lambda x: (2020 - x) if x > 0 else 0)


df_test = df_test.drop(['BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'], axis = 1)
for val in df_test[['BsmtQual', 'BsmtCond']]:
    df_test[val] = df_test[val].fillna('None')

df_test['MasVnrType'] = df_test['MasVnrType'].fillna('None')
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(0)

df_test['Electrical'] = df_test['Electrical'].fillna(df_test['Electrical'].mode()[0])
# %%
df_out_train = df
df_out_train.to_csv('train_cleaned.csv', index = False)

df_out_test = df_test
df_out_test.to_csv('test_cleaned.csv', index = False)
# %%

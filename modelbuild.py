import pandas as pd
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

#%%
X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')

X_test = pd.read_csv('X_test.csv')
#%%
# normalized data and log y_train
from sklearn.preprocessing import Normalizer
nor = Normalizer()
X_train_nor = nor.fit_transform(X_train)
X_test_nor = nor.transform(X_test)

Y_train_log = np.log(Y_train)
#%%
# converting dataframe to tensor
X_train_tensor = torch.tensor(X_train.to_numpy())
Y_train_tensor = torch.tensor(Y_train_log.values)
X_test_tensor = torch.tensor(X_test.to_numpy())
# building neural network

# plot loss function
# output the prediction
# %%

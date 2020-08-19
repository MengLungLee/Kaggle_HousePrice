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

Y_train = np.log(Y_train)
#%%
# converting dataframe to tensor

# building neural network
# plot loss function
# output the prediction
# %%

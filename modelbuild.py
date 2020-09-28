import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch
import matplotlib.pyplot as plt
import imageio
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = pd.read_csv('X_train.csv')
Y_train = pd.read_csv('Y_train.csv')

X_test = pd.read_csv('X_test.csv')

#%%
# normalized data and log y_train
from sklearn.preprocessing import Normalizer
nor = Normalizer()
X_train_nor = nor.fit_transform(X_train)
Y_train_log = np.log(Y_train)

X_test_nor = nor.transform(X_test)
# building neural network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()

        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, 100)
        self.hidden3 = torch.nn.Linear(100, 50)
        self.hidden4 = torch.nn.Linear(50, 25)
        self.predict = torch.nn.Linear(25, n_output)
    
    def forward(self, x):
        
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)
        return x

model = Net(n_feature = 50, n_hidden = 200, n_output = 1).to(device)


#%%
# converting dataframe to tensor and build up dataset with torch
X_train_tensor = torch.from_numpy(X_train.to_numpy())
Y_train_tensor = torch.from_numpy(Y_train_log.values)

X_test_tensor = torch.from_numpy(X_test.to_numpy())

torch_dataset = Data.TensorDataset(X_train_tensor, Y_train_tensor)

batch_size = 120

loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers = 2,
)
#%%
# training whole dataset with validation setting process

def RMSEloss(y, prediction):
    return torch.sqrt(torch.mean((prediction - y) **2))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

total_step = len(loader)
total_epoch = 1000
model.train()
for epoch in range(total_epoch):
    for step, (batch_x, batch_y) in enumerate(loader):
        
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        prediction = model(batch_x.float())
        train_loss = RMSEloss(batch_y, prediction)
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
#%% 
# applying training model on testset
with torch.no_grad():
    model.eval()
    output = model(X_test_tensor.float())

# # %%
# # output the prediction
print (output)
print (max(np.exp(output.numpy())) - min(np.exp(output.numpy())))
submission = pd.read_csv('sample_submission.csv')
submission['SalePrice'] = np.exp(output.numpy())
submission.to_csv('submission1000_a.csv', index=False)
print('RMSLE score on train data:{:.4f}' .format(RMSEloss(batch_y, prediction)))
# %%

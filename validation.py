import pandas as pd
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import torch
import matplotlib.pyplot as plt

#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train = pd.read_csv('X_train.csv', index_col = False)
Y_train = pd.read_csv('Y_train.csv', index_col = False)

#%%
# normalized data and log y_train
from sklearn.preprocessing import Normalizer
nor = Normalizer()
X_train_nor = nor.fit_transform(X_train)
Y_train_log = np.log(Y_train)
#%%
# split training set to validation set 
from sklearn.model_selection import train_test_split
Xv_train, Xv_validation, yv_train, yv_validation = train_test_split(X_train_nor, Y_train_log, test_size=0.3, random_state=0)

Xv_train_tensor = torch.from_numpy(Xv_train)
Xv_validation_tensor = torch.from_numpy(Xv_validation)
yv_train_tensor = torch.from_numpy(yv_train.values).view(-1,1)
yv_validation_tensor = torch.from_numpy(yv_validation.values).view(-1,1)

torch_dataset = Data.TensorDataset(Xv_train_tensor, yv_train_tensor)
#%%
# construct dataloader
batch_size = 120

loader = Data.DataLoader(
    dataset = torch_dataset,
    batch_size = batch_size,
    shuffle = True,
    num_workers= 2
)
#%%
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

def RMSEloss(y, prediction):
    return torch.sqrt(torch.mean((prediction - y) **2))

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_loss_list, val_loss_list = [], []

model.train()
total_step = len(loader)

for epoch in range(1000):
    train_loss = 0
    for step, (batch_x, batch_y) in enumerate(loader):
        
        optimizer.zero_grad()
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        prediction = model(batch_x.float())
        train_loss = RMSEloss(batch_y, prediction)
        
        train_loss.backward()
        optimizer.step()
    else:
        val_loss = 0
        with torch.no_grad():
            model.eval()
            output = model(Xv_validation_tensor.float())
            val_loss += RMSEloss(yv_validation_tensor, output)

        train_loss_list.append(train_loss/len(loader))
        val_loss_list.append(val_loss)
        print("Epoch: {}/{}.. ".format(epoch+1, 1000),
              "Training Loss: {:.3f}.. ".format(train_loss/len(loader)),
              "Validation Loss: {:.3f}.. ".format(val_loss))

plt.plot(train_loss_list, label = 'Training loss')
plt.plot(val_loss_list, label = 'Validation loss')
plt.legend(frameon=False)
print("Difference between Max SalePrice and Min SalePrice {}" .format(max(np.exp(output.numpy())) - min(np.exp(output.numpy()))))

#%%
# RMSE of training data set
print('RMSLE score on train data:{:.4f}' .format(RMSEloss(batch_y, prediction)))
print('RMSLE score on Validation data:{:.4f}' .format(RMSEloss(yv_validation_tensor, output)))
#%%

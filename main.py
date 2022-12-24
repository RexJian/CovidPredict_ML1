import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
config={
    'random_state':20,
    'train_size':0.8,
    'bastch_size':256,
    'learning_rate':0.001,
    'epoches':20000
}


class Covid19DataSet(Dataset):

    def __init__(self, x, y=None):
        if y is None:
            self.y = y
        else:
            self.y = torch.Tensor(np.array(y))
        self.x = torch.Tensor(np.array(x))

    def __getitem__(self, item):
        if self.y is None:
            return self.x[item]
        else:
            return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)
train_df=pd.read_csv('D:/PythonProjects/MachineLeaning/CovidPredict_ML1/Data/covid_train.csv')
test_df=pd.read_csv('D:/PythonProjects/MachineLeaning/CovidPredict_ML1/Data/covid_test.csv')

x=train_df[train_df.columns[1:94]]
y=train_df[train_df.columns[94]]
z=test_df[test_df.columns[1:94]]

x_train,x_valid,y_train,y_valid=train_test_split(x,y,train_size=config['train_size'],random_state=config['random_state'])

train_dataset=Covid19DataSet(x_train,y_train)
valid_dataset=Covid19DataSet(x_valid,y_valid)
test_dataset=Covid19DataSet(z)

train_DataLoader=DataLoader(train_dataset,batch_size=config['bastch_size'])
valid_DataLoader=DataLoader(valid_dataset,batch_size=config['bastch_size'])
test_DataLoader=DataLoader(test_dataset,batch_size=config['bastch_size'])


class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.Linear(8, 1)
        )
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = self.model(x)
        x=x.squeeze(1)
        return x

    def cal_loss(self,outputs, targets):
        train_loss = self.loss.forward(outputs, targets)
        return train_loss


def train_valid_optimize(input_dim, train_dataloader, valid_dataloader, epoches, learn_rate):
    model = Model(input_dim)
    model = model.to(device)
    model.loss = model.loss.to(device)
    optim = torch.optim.Adam(model.parameters(), learn_rate)
    writer=SummaryWriter("logs")
    step=0
    for epoch in range(epoches):
        model.train()
        total_train_loss = 0.0
        total_valid_loss = 0.0
        for data in train_dataloader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = model.cal_loss(outputs, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_train_loss += loss

        writer.add_scalar('TrainLoss',total_train_loss,step)
        model.eval()
        with torch.no_grad():
            for data in valid_dataloader:
                inputs,targets=data
                inputs=inputs.to(device)
                targets=targets.to(device)
                outputs=model(inputs)
                loss=model.cal_loss(outputs,targets)
                total_valid_loss+=loss
        writer.add_scalar('ValidLoss',total_valid_loss,step)
        step+=1
        print("{}".format(step)+"TrainLoss: {}".format(total_train_loss)+"   ValidLoss: {}".format(total_valid_loss))
        torch.save(model.state_dict(),"covid_model.pth")
    writer.close()


TrainValid=train_valid_optimize(x_train.shape[1],train_DataLoader,valid_DataLoader,config["epoches"],config["learning_rate"])

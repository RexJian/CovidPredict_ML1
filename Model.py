import torch.nn as nn
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
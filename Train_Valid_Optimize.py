from torch.utils.tensorboard import SummaryWriter

from Optimize import *

from Model import Model

class Train_Valid_Optimize:

    def __init__(self,input_dim, train_dataloader, valid_dataloader, epoches, learn_rate,device,model_name):
        self.input_dim=input_dim
        self.train_dataloader=train_dataloader
        self.valid_dataloader=valid_dataloader
        self.epoches=epoches
        self.learn_rate=learn_rate
        self.device=device
        self.model_name=model_name

    def train_valid_optimize(self):
        model = Model(self.input_dim)
        model = model.to(self.device)
        model.loss = model.loss.to(self.device)
        optim = torch.optim.Adam(model.parameters(), self.learn_rate)
        valid=Valid(model,self.valid_dataloader,self.device)
        writer=SummaryWriter("logs")
        step=0
        for epoch in range(self.epoches):
            model.train()
            total_train_loss = 0.0
            for data in self.train_dataloader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = model(inputs)
                loss = model.cal_loss(outputs, targets)
                optim.zero_grad()
                loss.backward()
                optim.step()
                total_train_loss += loss
            writer.add_scalar('TrainLoss', total_train_loss, step)
            total_valid_loss=valid.Valid()
            writer.add_scalar('ValidLoss',total_valid_loss, step)
            step+=1
            print("{}".format(step)+"TrainLoss: {}".format(total_train_loss)+"   ValidLoss: {}".format(total_valid_loss))
            torch.save(model.state_dict(),self.model_name+".pth")
            writer.close()
import torch


class Valid:

    def __init__(self,model,valid_dataloader, device):

        self.valid_dataloader = valid_dataloader
        self.device = device
        self.model=model

    def Valid(self,total_valid_loss=0):
        self.model.eval()
        with torch.no_grad():
            for data in self.valid_dataloader:
                inputs, targets = data
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.model.cal_loss(outputs, targets)
                total_valid_loss += loss

        return  total_valid_loss


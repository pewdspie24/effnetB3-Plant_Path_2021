import torch
import torchsummary
import torch.nn as nn
import copy
from efficientnet_pytorch import EfficientNet

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def accuracy(predictions, labels):
    classes = torch.argmax(predictions, dim=1)
    return torch.mean((classes == labels).float())

class Model(nn.Module):
    @staticmethod
    def out(in_channels):
        return nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.Linear(in_channels, 512),
            nn.SiLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512,128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,12)
        )
    def __init__(self):
        super().__init__()
        self.modelPre = EfficientNet.from_pretrained('efficientnet-b3')
        self.num_fl = self.modelPre._fc.in_features
        self.model = copy.deepcopy(self.modelPre)
        self.model._fc = Model.out(self.num_fl)
    
    def forward(self, x):
        return self.model(x)
    
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad(False)
        for param in self.model._fc.parameters():
            param.requires_grad(True)
        
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad(True)
    
    def summ(self,i):
        torchsummary.summary(self.model,input_size=i)
        return
        
if __name__ == "__main__":
    model = Model().cuda()
    torchsummary.summary(model, input_size=(3,600,600))
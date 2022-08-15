import torch
from torch import nn




class ACE_Net(nn.Module):
    def __init__(self):
        super(ACE_Net, self).__init__() 
        self.features = nn.Sequential(
            
            nn.Conv1d(in_channels=1,out_channels=32, kernel_size=10),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, stride=1),
            nn.Dropout(p = 0.4),
            nn.Conv1d(in_channels=16,out_channels=8, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(in_channels=8, out_channels=4, kernel_size=5),
            nn.ReLU(inplace=True),
        
        )
              
        self.fatten = nn.Flatten()
        self.linear1 = nn.Linear(264,132)
        self.linear2 = nn.Linear(132, 66)
        self.linear3 = nn.Linear(66, 3)
        self.linear4 = nn.Linear(3,1)
        
        
    def forward(self, x):
        x = self.features(x)
        x = self.fatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        
        return x       
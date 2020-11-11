import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
# from torchsummary import summary

class TLClassification(nn.Module):
    def __init__(self):

        super(TLClassification, self).__init__()
        
        # self.backbone = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
        # for param in model.parameters():
        #     param.requires_grad = False
        Dout = 0.2
        n_class = 3
        
        inp = (3, 40, 40) 

        self.cnn_layers = nn.Sequential(
            
            nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(4 * 10 * 10, 3)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
        

if __name__ == "__main__":
    
    model = TLClassification()
    mpdel = model.cuda()

    summary(model, (3, 40, 40))
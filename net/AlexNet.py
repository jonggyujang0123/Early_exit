import torch.nn as nn
import torch
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self,num_classes=10):
        a=0
        
        
class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        a=0

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet,self).__init__()

        self.net_part1 = nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2), 
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=3, stride=2), 
          nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75),
          )
        self.classifier_1 = nn.Sequential(
          nn.ReLU(),
          nn.MaxPool2d(3, 2), 
          nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75),
          nn.Conv2d(64, 32, 3, 1, 1), 
          nn.ReLU(),
          nn.Conv2d(32, 32, 3, 1, 1), 
          nn.ReLU(),
          nn.MaxPool2d(3, 2), 
          nn.Flatten(),
          nn.Linear(in_features=288, out_features=num_classes),
          )
        self.net_part2 = nn.Sequential(
          nn.Conv2d(64, 96, kernel_size=5, stride=1, padding=2), 
          nn.ReLU(),
          nn.MaxPool2d(3, 2), 
          nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75),
          nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=1), 
          )
        
        self.classifier_2 = nn.Sequential(
          nn.ReLU(),
          nn.MaxPool2d(3, 2), 
          nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75),
          nn.Conv2d(192, 64, kernel_size=3, stride=1, padding=1), 
          nn.ReLU(),
          nn.MaxPool2d(3, 2), 
          nn.Flatten(),
          nn.Linear(in_features=64, out_features=num_classes),
          )
        
        self.net_part3 = nn.Sequential(
          nn.ReLU(),
          nn.Conv2d(192, 128, 3, 1, 1), 
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, 1, 1), 
          nn.ReLU(),
          nn.MaxPool2d(3, 2),
          nn.Flatten(),
          nn.Linear(in_features=576*2, out_features=1024),
          nn.ReLU(),
          nn.Linear(in_features=1024, out_features=1024),
          nn.ReLU(),
          nn.Linear(in_features=1024, out_features=num_classes),
          )
    
    def forward(self,x):
        x = self.net_part1(x)
        out_1 = self.classifier_1(x)
        x = self.net_part2(x)
        out_2 = self.classifier_2(x)
        out_3 = self.net_part3(x)

        return out_1, out_2, out_3





def main():
    return 

if __name__ == '__main__':
    main()
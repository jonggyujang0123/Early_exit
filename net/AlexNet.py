import torch.nn as nn
import torch
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self,num_classes=10, in_channel = 1):
        super(LeNet,self).__init__()
        
        self.net_part1 = nn.Sequential(
            nn.Conv2d(in_channel, 5, kernel_size= 5, stride= 1, padding =3),
            nn.MaxPool2d(2,2),
            nn.ReLU()) ## 1, 5, 34, 34
        self.classifier1 = nn.Sequential(
            nn.Conv2d(5, 10, 3, 1, padding = 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features = 640, out_features = num_classes)
            )
        self.net_part2 = nn.Sequential(
            nn.Conv2d(5, 10, 5, 1, 3),
            nn.MaxPool2d(2,2),
            nn.ReLU()
            )
        self.classifier2 = nn.Sequential(
            nn.Conv2d(10, 20, 5, 1, 1),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=180, out_features= num_classes))
        self.net_part3 = nn.Sequential(
            nn.Conv2d(10,20,5,1,3),
            nn.MaxPool2d(2,2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=500, out_features= 84),
            nn.Linear(in_features=84, out_features= num_classes)
            )
            
    def forward(self,x):
        x = self.net_part1(x)
        out_1 = self.classifier_1(x)
        x = self.net_part2(x)
        out_2 = self.classifier_2(x)
        out_3 = self.net_part3(x)

        return out_1, out_2, out_3        
        
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

                                                                                                                                                                              
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channel = 1):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(in_channel, 16, kernel_size=5,
                               stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.classifier1 = nn.Sequential(
            nn.Conv2d(16, 64, 5, 1, 2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32768, num_classes)
            )
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.classifier2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            nn.Linear(12544, num_classes)
            )
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.linear = nn.Linear(128*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.classifier1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out2 = self.classifier2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out3 = self.linear(out)
        return out1, out2, out3


def ResNet18(num_classes, in_channel):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, in_channel =in_channel)
        

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, in_channel = 1):
        super(AlexNet,self).__init__()

        self.net_part1 = nn.Sequential(
          nn.Conv2d(in_channel, 64, kernel_size=5, stride=1, padding=2), 
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

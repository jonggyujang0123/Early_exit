import numpy as np
import torch
from torchvision import transforms, datasets
import ssl
#certification error
ssl._create_default_https_context = ssl._create_unverified_context
 

data_name = 'CIFAR10'
net_name = 'AlexNet'


train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.Resize(256),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229,0.224,0.225])
                                      ])


train_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229,0.224,0.225])
                                      ])

data_train = datasets.CIFAR10(root= './DATA', train= True,
                              download = True, transform=train_transform)


def main():
    return
    
    
    
if __name__ == '__main__':
    main()
    

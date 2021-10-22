# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 20:17:53 2021

@author: jongg
"""
from torchvision import transforms, datasets
import torch
import matplotlib.pyplot as plt
import numpy as np
## FUNCTIONS

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    return 


def dataloader(data_name, batch_size):
    if data_name == 'CIFAR10' or data_name == 'CIFAR100':
        train_transform = transforms.Compose([transforms.ToTensor(), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.Resize(32),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229,0.224,0.225])
                                      
                                      ])
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Resize(32),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229,0.224,0.225])
                                      ])
        in_channel = 3
    elif data_name == 'MNIST' or data_name == 'EMNIST':
        train_transform = transforms.Compose([transforms.ToTensor(), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.Resize(32),
                                      transforms.Normalize((0.485,), (0.225,))
                                      
                                      ])
        test_transform = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Resize(32),
                                      transforms.Normalize((0.485,), (0.225,))
                                      ])
        in_channel =1
    if data_name == 'CIFAR10':
        data_train = datasets.CIFAR10(root= './DATA', train= True,
                                  download = True, transform=train_transform)
        data_test = datasets.CIFAR10(root= './DATA', train= False,
                                 download = True, transform=test_transform)
        class_num = 10
    elif data_name == 'CIFAR100':
        data_train = datasets.CIFAR10(root= './DATA', train= True,
                                  download = True, transform=train_transform)
        data_test = datasets.CIFAR10(root= './DATA', train= False,
                                 download = True, transform=test_transform)
        class_num = 100
    elif data_name == 'MNIST':
        data_train = datasets.MNIST(root= './DATA', train= True,
                                  download = True, transform=train_transform)
        data_test = datasets.MNIST(root= './DATA', train= False,
                                 download = True, transform=test_transform)
        class_num= 10
    elif data_name == 'EMNIST':
        data_train = datasets.EMNIST(root= './DATA', train= True, split = 'bymerge',
                                  download = True, transform=train_transform)
        data_test = datasets.EMNIST(root= './DATA', train= False, split = 'bymerge',
                                 download = True, transform=test_transform)
        class_num = 47
    else:
        raise ValueError('Choose appropriate dataset (CIFAR10, CIFAR100, MNIST, EMNIST)')
    trainloader = torch.utils.data.DataLoader(data_train, batch_size= batch_size, shuffle = True, num_workers=1)
    testloader = torch.utils.data.DataLoader(data_test, batch_size= batch_size, shuffle = False, num_workers=1)
    return trainloader, testloader, class_num, in_channel

def main():
    return


if __name__ == '__main__':
    main()

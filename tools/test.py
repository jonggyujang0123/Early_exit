import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import ssl
import matplotlib.pyplot as plt
from net.AlexNet import AlexNet, LeNet, ResNet18
from tqdm import tqdm
from tools.dataloader import dataloader
import os
import torchprof
import argparse
#certification error
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dataset',
                   help='Choose among MNIST, CIFAR10, CIFAR100, EMNIST')
parser.add_argument('--model',
                   help='Choose among AlexNet, LeNet, ResNet')
parser.add_argument('--th1', type = float, help = 'first threshold')
parser.add_argument('--th2', type = float, help = 'first threshold')

args = parser.parse_args()
 

data_name = args.dataset
net_name = args.model

thr=  np.array([args.th1, args.th2]).reshape([1,2])

batch_size=256

trainloader, testloader, num_classes, in_channel = dataloader(data_name = data_name,batch_size = batch_size)


## NETWORK LOADER
if net_name == 'AlexNet':
    Net = AlexNet(num_classes = num_classes, in_channel =in_channel)
    lr = 0.006
elif net_name == 'LeNet':
    Net = LeNet(num_classes = num_classes, in_channel =in_channel)
    lr = 0.006
elif net_name == 'ResNet':
    Net = ResNet18(num_classes = num_classes, in_channel =in_channel)
    lr = 0.001
else:
    raise ValueError('Choose correct model')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

## MAIN FUNCTION


def main():
    acc_list = list()
    exit_point_list = list()
    o_acc_list = list()
    for batch_idz, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output_1, output_2, output_3 = Net.forward(inputs)
        o_output =  np.concatenate([np.expand_dims(output_1.detach().cpu().numpy(),axis=1),
                                       np.expand_dims(output_2.detach().cpu().numpy(),axis=1),
                                       np.expand_dims(output_3.detach().cpu().numpy(),axis=1)],
                                      axis=1)
        o_output = np.exp(o_output) / np.sum(np.exp(o_output),axis=2,keepdims=True)
        entropy = - np.sum(o_output * np.log(o_output+1e-5),axis=2)[:,0:2]
        threshold = entropy < thr
        exit_point = np.concatenate( (threshold[:,0:1], (1-threshold[:,0:1])*threshold[:,1:2], (1-threshold[:,0:1]) * (1-threshold[:,1:2])   ), axis=1)
        #exit_point = (exit_point == np.arange(3).reshape([1,3])) +0.0

        inference = np.concatenate([np.expand_dims(torch.argmax(output_1.detach().cpu(),axis=1).numpy(),axis=1),
                                       np.expand_dims(torch.argmax(output_2.detach().cpu(),axis=1).numpy(),axis=1),
                                       np.expand_dims(torch.argmax(output_3.detach().cpu(),axis=1).numpy(),axis=1)],
                                      axis=1)
        correct = (inference == labels.detach().cpu().numpy().reshape([-1,1]))
        o_correct = np.sum((inference == labels.detach().cpu().numpy().reshape([-1,1]))*exit_point,axis=1,keepdims=True)
        acc_list.append(correct)
        o_acc_list.append(o_correct)
        exit_point_list.append(exit_point)
    with torchprof.Profile(Net,use_cuda=True, profile_memory=True) as prof:
        Net(inputs)
    print(prof.display(show_events=False))
    acc = np.concatenate(acc_list,axis=0)
    o_acc = np.concatenate(o_acc_list,axis=0)
    exit_point = np.concatenate(exit_point_list,axis=0)
    print(f'|Accuracy| \n 1st output is {np.mean(acc,axis=0)[0]} \n 2nd output is {np.mean(acc,axis=0)[1]} \n 3rd output is {np.mean(acc,axis=0)[2]}, \n overall is {np.mean(o_acc)} \n exit_point is {np.mean(exit_point,axis=0)}' )
    return
    
    
    
if __name__ == '__main__':
    if os.path.exists(f'./DATA/{data_name}_{net_name}.pth'):
        Net.load_state_dict(torch.load(f'./DATA/{data_name}_{net_name}.pth'))
        Net.to(device)
    main()

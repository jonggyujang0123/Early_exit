import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import ssl
import matplotlib.pyplot as plt
from net.AlexNet import AlexNet
from tqdm import tqdm
from tools.dataloader import dataloader
import os
import torchprof


#certification error
ssl._create_default_https_context = ssl._create_unverified_context
 

data_name = 'CIFAR10'
net_name = 'AlexNet'

batch_size=300

_, testloader = dataloader(data_name = data_name,batch_size = batch_size)

## NETWORK LOADER
if net_name == 'AlexNet':
    Net = AlexNet()
elif net_name == 'b':
    print(100)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Net.to(device)
criterion = nn.CrossEntropyLoss()

## MAIN FUNCTION


def main():
    acc_list = list()
    for batch_idz, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        output_1, output_2, output_3 = Net.forward(inputs)
        inference = np.concatenate([np.expand_dims(torch.argmax(output_1.detach().cpu(),axis=1).numpy(),axis=1),
                                       np.expand_dims(torch.argmax(output_2.detach().cpu(),axis=1).numpy(),axis=1),
                                       np.expand_dims(torch.argmax(output_3.detach().cpu(),axis=1).numpy(),axis=1)],
                                      axis=1)
        correct = (inference == labels.detach().cpu().numpy().reshape([-1,1]))
        acc_list.append(correct)
    with torchprof.Profile(Net,use_cuda=True, profile_memory=True) as prof:
        Net(inputs)
    print(prof.display(show_events=False))
    acc = np.concatenate(acc_list,axis=0)
    print(f'|Accuracy| \n 1st output is {np.mean(acc,axis=0)[0]} \n 2nd output is {np.mean(acc,axis=0)[1]} \n 3rd output is {np.mean(acc,axis=0)[2]}' )
    return
    
    
    
if __name__ == '__main__':
    if os.path.exists(f'./DATA/{data_name}_{net_name}.pth'):
        torch.load(f'./DATA/{data_name}_{net_name}.pth')
    main()
import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import ssl
import matplotlib.pyplot as plt
from net.AlexNet import AlexNet, LeNet, ResNet18
from tqdm import tqdm
from tools.dataloader import dataloader
import argparse
#certification error
ssl._create_default_https_context = ssl._create_unverified_context

parser = argparse.ArgumentParser(description = 'Options')
parser.add_argument('--dataset',
                   help='Choose among MNIST, CIFAR10, CIFAR100, EMNIST')
parser.add_argument('--model',
                   help='Choose among AlexNet, LeNet, ResNet')
args = parser.parse_args()
 

data_name = args.dataset
net_name = args.model

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
Net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Net.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

## MAIN FUNCTION

def main():
    epoches = tqdm(range(200))
    for epoch in epoches:
        loss_ep = list()
        for batch_idz, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            output_1, output_2, output_3 = Net.forward(inputs)
            loss = 1.0 * criterion(output_1, labels) + 0.4 * criterion(output_2, labels) + 0.3 * criterion(output_3, labels)
            loss.backward()
            optimizer.step()
            loss_ep.append(loss.item())
        scheduler.step()
        if epoch % 20 == 19:
            tqdm.write(f'{epoch}-th epoch loss is {np.mean(loss_ep)}')
            torch.save(Net.state_dict(), f'./DATA/{data_name}_{net_name}.pth')
    return
    
    
    
if __name__ == '__main__':
    main()
    

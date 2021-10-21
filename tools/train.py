import numpy as np
import torch
from torchvision import transforms, datasets
import torch.nn as nn
import ssl
import matplotlib.pyplot as plt
from net.AlexNet import AlexNet
from tqdm import tqdm
#certification error
ssl._create_default_https_context = ssl._create_unverified_context
 

data_name = 'CIFAR10'
net_name = 'AlexNet'

batch_size=300

## DATA LOADER
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

data_train = datasets.CIFAR10(root= './DATA', train= True,
                              download = True, transform=train_transform)
trainloader = torch.utils.data.DataLoader(data_train, batch_size= batch_size, shuffle = True, num_workers=1)

data_test = datasets.CIFAR10(root= './DATA', train= False,
                             download = True, transform=test_transform)
testloader = torch.utils.data.DataLoader(data_test, batch_size= batch_size, shuffle = False, num_workers=1)


classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

## NETWORK LOADER

Net = AlexNet()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
## FUNCTIONS

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

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
        if epoch % 20 == 19:
            tqdm.write(f'{epoch}-th epoch loss is {np.mean(loss_ep)}')
    torch.save(Net, f'./DATA/{data_name}_{net_name}.pth')
    return
    
    
    
if __name__ == '__main__':
    main()
    

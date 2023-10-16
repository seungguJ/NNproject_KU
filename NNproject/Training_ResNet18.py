import torch
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from ResNet import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

transform = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
    transforms.ToTensor(),
])

trainset = torchvision.datasets.CIFAR10(root='...data_path...', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='...data_path...', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=64, shuffle=False)

model = ResNet18()
model.to(device)

critertion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1,
                    momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

num_epochs = 200
loss_ = []
n = len(trainloader)

model.train()

for epoch in range(num_epochs):
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        
        loss = critertion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    loss_.append(running_loss/n)
    print(f'{epoch+1} loss: {running_loss/n}')
    scheduler.step()
    
# prediction part
correct = 0
total = 0
with torch.no_grad(): 
  model.eval() 
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1) 
    total += labels.size(0) 
    correct += (predicted == labels).sum().item()

print(f'accuracy of 10000 test images: {100*correct/total}%')

PATH = '...checkpoint_path...'
torch.save(model, PATH + 'resnet18_cifar10.pt')

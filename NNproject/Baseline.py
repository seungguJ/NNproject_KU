import torch
import torchvision.transforms as transforms
import torchvision
from ResNet import *

def attack(imgs, l2 = True, eps = 1, device=False):
    if l2:
        noise = torch.normal(mean = 0., std = 0.05, size=(imgs.shape[0], 3, 32, 32)).to(device)
        scaling_factor  = torch.norm(noise, p=2)/(3*(imgs.shape[0]**(1/2))) # scaling the noise because of the l2 norm getting bigger as the size of input increases
        noise *= (eps / scaling_factor)
        adv_imgs = imgs + noise
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
    else:
        noise = torch.normal(mean = 0., std = 0.05, size=(imgs.shape[0], 3*32*32)).to(device)
        noise = torch.clamp(noise, -eps, eps)
        noise = noise.view(imgs.shape[0], 3, 32, 32)
        adv_imgs = imgs + noise
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
    return adv_imgs
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

transform = transforms.Compose([
    transforms.ToTensor(),
])

batch_size = 100

testset = torchvision.datasets.CIFAR10(root='...data_path...', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset,batch_size=batch_size, shuffle=False)

model = ResNet18() ## original model
checkpoint_path2 = './resnet18_cifar10'
checkpoint = torch.load(checkpoint_path2)['net']
for key in list(checkpoint.keys()):
    checkpoint[key.replace('module.', '')] = checkpoint.pop(key)
model.load_state_dict(checkpoint, strict=False)## original model
model = model.to(device)
model.eval()

correct, adv_correct = 0, 0
total = 0
with torch.no_grad(): # Deactivate gradient calculation
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    adv_images = attack(images, device=device) # For l2 norm attack
    # adv_imges = attack(imges,l2=False, eps=0.1 device=device) # For infinity norm attack
    outputs = model(images)
    adv_outputs = model(adv_images)
    _, predicted = torch.max(outputs.data, 1) # Probability, index
    _, adv_predicted = torch.max(adv_outputs.data, 1) # Probability, index
    total += labels.size(0) # Number of labels
    correct += (predicted == labels).sum().item() # Number of correct predictions
    adv_correct += (adv_predicted == labels).sum().item() # Number of correct predictions

print(correct/total, adv_correct/total) # Accuracy

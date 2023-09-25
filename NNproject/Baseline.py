import torch
import torchvision.transforms as transforms
import torchvision

def attack(imgs, l2 = True, eps = 8/255, device=False):
    if l2:
        noise = torch.normal(mean = 0., std = 0.1, size=(imgs.shape[0],3*32*32)).to(device)
        condi = abs(eps)**0.5
        noise = torch.where(noise > condi, condi, noise)
        noise = torch.where(noise < -condi, -condi, noise)
        noise = noise.view(imgs.shape[0], 3, 32, 32)
        adv_imgs = imgs + noise
        adv_imgs = torch.clamp(adv_imgs, 0, 1)
    else:
        noise = torch.normal(mean = 0., std = 0.1, size=(imgs.shape[0],3*32*32)).to(device)
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

model = torch.load('...model_path.../resnet18_cifar10.pt')
model = model.cuda()
model.eval()

correct, adv_correct = 0, 0
total, adv_total = 0, 0
with torch.no_grad(): # Deactivate gradient calculation
  model.eval() 
  for data in testloader:
    images, labels = data[0].to(device), data[1].to(device)
    adv_images = attack(images, device=device)
    # For infinity norm attack
    # adv_imges = attack(imges,l2=False, eps=0.05 device=device)
    outputs = model(images)
    adv_outputs = model(adv_images)
    _, predicted = torch.max(outputs.data, 1) # Probability, index
    _, adv_predicted = torch.max(adv_outputs.data, 1) # Probability, index
    total += labels.size(0) # Number of labels
    adv_total += labels.size(0) # Number of labels
    correct += (predicted == labels).sum().item() # Number of correct predictions
    adv_correct += (adv_predicted == labels).sum().item() # Number of correct predictions

print(correct/total, adv_correct/total) # Accuracy
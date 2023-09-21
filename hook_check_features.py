import torch
import torchvision.models as models

# For vgg
model = models.vgg19(pretrained=True)
model.eval().cuda()

holder = []        
        
def forwardhooking(m, inp, out):
    holder.append(out)
# forwardhooking = ForwardHooking

for name, m in model.named_modules():
    if name == 'features.33':
        m.register_forward_hook(forwardhooking)

model(torch.rand(1, 3, 224, 224).cuda())



# For resnet
model = models.resnet50(pretrained=True)
model.eval().cuda()

holder = []        
        
def forwardhooking(m, inp, out):
    holder.append(out)
# forwardhooking = ForwardHooking

for name, m in model.named_modules():
    if name == 'layer4.2':
        m.register_forward_hook(forwardhooking)

model(torch.rand(1, 3, 224, 224).cuda())

print(holder[0].shape)
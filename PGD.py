import torch
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image

class PGD():
    
    def __init__(self, model, cuda=False, eps=8/255, alpha=2/255, iters=20):
        self.cuda = cuda
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.model = model.cuda()
        self.model.eval()
    
    def __call__(self, images, labels):
        images = images.cuda()
        labels = labels.cuda()
        loss = nn.CrossEntropyLoss()
            
        ori_images = images.data
            
        for i in range(self.iters) :    
            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs, labels).cuda()
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
        # images = to_pil_image(images.cpu().squeeze())
            
        
        return images
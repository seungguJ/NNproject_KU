import torch
import torch.nn.functional as F

class Integrated_Gradients(object):
    
    def __init__(self, pretrained_model, cuda=False):
        self.cuda = cuda
        self.pretrained_model = pretrained_model.cuda()
    
    '''
    input with tensor [1, C, H, W]
    '''
    
    
    def __call__(self, input,target_index=None, steps=50,  random_path=False, baseline_tensor=None, layover=None):
        
        if steps % 2 != 0:
            steps += 1
        input_batch = input.cuda() # input with torch.Size([batch, C, H, W])
        # if type(self.pretrained_model) == models.inception.Inception3:
        #     _, output = self.pretrained_model(input_batch)
        # else:
        output = self.pretrained_model(input_batch)
        if target_index is None:
            target_index = output.argmax().item()
           
        h_x = F.softmax(output, dim=1).data.squeeze()
        target_prob = h_x[target_index]
        if baseline_tensor == None:
            baseline_tensor = torch.zeros_like(input_batch)
        baseline_tensor=baseline_tensor.cuda()
        
        if layover != None:
            layover = layover.cuda()
        
        total_gradients = torch.zeros_like(input_batch)
        if layover == None:
            for i in range(steps+1):
                if random_path and i != steps:
                    scaled_input = baseline_tensor + (i / steps* (input_batch-baseline_tensor))
                    scaled_input = scaled_input + torch.randn_like(scaled_input)*0.01
                else:
                    scaled_input = baseline_tensor + (i / steps * (input_batch-baseline_tensor))
                
        else:
            for i in range(steps+1):
                if random_path :
                    if i <= steps//2:
                        scaled_input = baseline_tensor + (i / (steps//2) * (layover-baseline_tensor))
                        scaled_input = scaled_input + torch.randn_like(scaled_input)*0.01
                    else:
                        if i != steps:
                            scaled_input = layover + ((i-steps//2) / (steps//2) * (input_batch-layover))
                            scaled_input = scaled_input + torch.randn_like(scaled_input)*0.01
                        else:
                            scaled_input = layover + ((i-steps//2) / (steps//2) * (input_batch-layover))
                else:
                    if i <= steps//2:
                        scaled_input = baseline_tensor + (i / (steps//2) * (layover-baseline_tensor))
                    else:
                            scaled_input = layover + ((i-steps//2) / (steps//2) * (input_batch-layover))
        
        scaled_input.requires_grad = True
        output = self.pretrained_model(scaled_input)
        loss = output[0,target_index]
        loss.backward()
        grad = scaled_input.grad.data
        total_gradients += grad/steps
            
        attr = (input_batch-baseline_tensor)*total_gradients
        # attr = attr.sum(dim=1, keepdim=True)
        # attr = F.relu(attr)
        # attr = (attr - torch.min(attr))/(torch.max(attr)-torch.min(attr))
        return target_index, target_prob, attr # predicted index and tensor with batch dimension [batch, 1, H, W]

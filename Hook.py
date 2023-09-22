class Hook():
    def __init__(self, module, backward=False):
        self.hook = module.register_backward_hook(self.forwardhooking) if backward else module.register_forward_hook(self.forwardhooking)
    def forwardhooking(self, module, input, output):
        self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


'''
Usage:

hookF = []
for name, m in model.named_modules():
    if "conv" in name or "bn" in name or "linear" in name: # 17 layers for conv and bn each and there is one linear layer
        hookF.append(Hook(m, backward=False))
        if name not in adv_cos_sim.keys():
            adv_cos_sim[name] = 0

model(ori_imgs) # The firt 35 layers are stored in holder
holder = [hook.output for hook in hookF]
model(pgd_imgs) # The last 35 layers are stored in holder
adv_holder = [hook.output for hook in hookF]

'''
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3):
        defaults = {"lr" : lr} 
        # these values get passed into self.param_groups as default parameters
        super().__init__(params, defaults) 
    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups: 
            lr = group["lr"] # get the learning rate.
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p] #get the state we are tracking
                t = state.get("t", 0 ) #get the iteration number or initialize
                grad = p.grad.data # grab the gradient computed by .backward()
                p.data -= lr * grad / (t+1)**0.5 #update the data 
                state["t"] = t + 1 #update the iteration number
        return loss

# when we use p.data to access the tensor, autograd does not modify the computation graph.
# even if we did this is still okay because even if they get modified we will 
# again call .backward() on loss to retrieve the graph
# more modern idiom us to access p directy inside context manager:
# with torch.no_grad

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas = (0.9, 0.999), weight_decay = 0.1, eps = 1e-8):
        defaults = {"lr":lr, "beta1":betas[0], "beta2":betas[1], "lambda_wd":weight_decay, "eps":eps}
        super().__init__(params, defaults)
    def step(self, closure = None):        
        loss = None if closure is None else closure() # API convention - not really used.
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p))
                v = state.get("v", torch.zeros_like(p))
                # correction to moving averages to betas are absorbed into learning rate
                lr_adj = group["lr"]*(1 - group["beta2"]**t)**0.5 / (1 - group["beta1"]**t)
                # using lr_adj vs lr is a choice -
                ## in place multiply: rather than p.data = p.data*(1-...)
                p.data.mul_(1 - group["lr"]*group["lambda_wd"]) # weight decay
                grad = p.grad.data
                state["m"] = group["beta1"]*m + (1 - group["beta1"])*grad
                state["v"] = group["beta2"]*v + (1 - group["beta2"])*grad**2
                p.data -= lr_adj * state["m"]/(state["v"]**0.5 + group["eps"])
                state["t"] = t + 1 #update the iteration number
        return loss
    


def lr_cosine_schedule(t,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters):
    if t < warmup_iters:
        return t * max_learning_rate / warmup_iters
    if t > cosine_cycle_iters:
        return min_learning_rate
    else:
        return min_learning_rate + (1 + math.cos((t - warmup_iters)*math.pi/(cosine_cycle_iters - warmup_iters)))*(max_learning_rate - min_learning_rate)/2.0


def gradient_clip(parameters, max_norm, eps = 1e-6):
    with torch.no_grad():
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.linalg.vector_norm(torch.concat([param.grad.flatten() for param in parameters]))
        if total_norm > max_norm:
            for param in parameters:
                param.grad.mul_(max_norm/(total_norm + eps))
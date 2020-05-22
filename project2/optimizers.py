import torch

class SGD_opti(object):
    def __init__(self, model_parameters, learn_rate = 1e-3,beta = 0.9):
        self.lr = learn_rate
        self.param_to_update= [{'param': p,'update': torch.zeros_like(p.data)} for p in model_parameters if p.data is not None]
        self.beta = beta
        self.steps = 0

    def step(self):
        for p in self.param_to_update:
            if p['param'].data is not None:
                if self.steps ==0:
                    p['update'] = p['param'].grad
                else:
                    p['update'] = self.beta * p['update'] + (1-self.beta)*p['param'].grad
                p['param'].data -= self.lr*p['update']
        self.steps+=1


    def zero_grad(self):
        for p in self.param_to_update:
            p['param'].input = None
            p['param'].grad = None



class Adam_opti(object):
    def __init__(self, model_parameters, learn_rate = 1e-3,beta1 = 0.9,beta2 = 0.999, eps = 1e-8):
        self.lr = learn_rate
        self.param_to_update= [{'param': p,'m': torch.zeros_like(p.data), 'v': torch.zeros_like(p.data)} for p in model_parameters if p.data is not None]
        self.beta1 = beta1
        self.steps = 0
        self.beta2 = beta2
        self.eps = eps
    
    def step(self):
        for p in self.param_to_update:
            if p['param'].data is not None:
                p['m'] = self.beta1 * p['m'] + (1-self.beta1)*p['param'].grad
                p['v'] = self.beta2 * p['v'] + (1-self.beta2)*p['param'].grad.pow(2)
                p['param'].data -= self.lr*(p['m']*(1- self.beta2)/((1-self.beta1) *(p['v'] + (1- self.beta2)*self.eps)))
        self.steps+=1
        
                
    def zero_grad(self):
        for p in self.param_to_update:
            p['param'].input = None
            p['param'].grad = None


class LR_scheduler(object):
    def __init__(self, optimizer, gamma=0.1,counter = 30):
        self.optimizer = optimizer
        self.gamma = gamma
        self.counter = counter
        self.steps = 0
        self.curr_count = counter

    def step(self):
        self.steps+=1
        if self.steps > self.curr_count:
            self.optimizer.lr *= self.gamma
            self.curr_count += self.counter





import torch

class SGD_opti(object):
    #SGD optimizer with momentum
    def __init__(self, model_parameters, learn_rate = 1e-3,beta = 0.9):
        self.lr = learn_rate
        #Stores model parameters to update with an additional update variable
        self.param_to_update= [{'param': p,'update': torch.zeros_like(p.data)} for p in model_parameters if p.data is not None]
        #beta is the parameter for Momentum
        self.beta = beta
        self.steps = 0

    def step(self):
        for p in self.param_to_update:
            if p['param'].data is not None:
                if self.steps ==0:
                    #For first step of momentum use the complete gradient 
                    p['update'] = p['param'].grad
                else:
                    #For subsequent steps apply momentum update
                    p['update'] = self.beta * p['update'] + (1-self.beta)*p['param'].grad
                #Update the weights
                p['param'].data -= self.lr*p['update']
        #Increase step counter
        self.steps+=1


    def zero_grad(self):
        #Zero grad function to forget gradients
        for p in self.param_to_update:
            #Update input and gradient for all parameters to None
            p['param'].input = None
            p['param'].grad = None



class Adam_opti(object):
    #Adam optimizers
    def __init__(self, model_parameters, learn_rate = 1e-3,beta1 = 0.9,beta2 = 0.999, eps = 1e-8):
        self.lr = learn_rate
        #Store first and second moments of gradient along with parameter
        self.param_to_update= [{'param': p,'m': torch.zeros_like(p.data), 'v': torch.zeros_like(p.data)} for p in model_parameters if p.data is not None]
        # Store Adam parameters beta1, beta2 and epsilon
        self.beta1 = beta1
        self.steps = 0
        self.beta2 = beta2
        self.eps = eps
    
    def step(self):
        for p in self.param_to_update:
            if p['param'].data is not None:
                #Update first and second moments of gradient 
                p['m'] = self.beta1 * p['m'] + (1-self.beta1)*p['param'].grad
                p['v'] = self.beta2 * p['v'] + (1-self.beta2)*p['param'].grad.pow(2)
                #Update the weights using bias corrected first and second moments
                p['param'].data -= self.lr*(p['m']*(1- self.beta2)/((1-self.beta1) *(p['v'] + (1- self.beta2)*self.eps)))
        #Increase step counter
        self.steps+=1
        
                
    def zero_grad(self):
        #Forget previous gradients
        for p in self.param_to_update:
            p['param'].input = None
            p['param'].grad = None




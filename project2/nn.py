import math
import torch

#Structure is better if original

class Module ( object ) :

    def forward ( self , * input ) :
        raise NotImplementedError
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    def param ( self ) :
        return self.params



class Parameter(object):
    def __init__(self, data, grad=0, x=None ):
        super(Parameter, self).__init__()
        self.data = data
        self.grad = grad
        self.input = x


class lossMSE(Module):
    def __init__(self):
        super(lossMSE, self).__init__()
        self.name = 'MSE_loss'
    def forward(self, x, target):
        return x.sub(target).pow(2).mean()
    def backward(self, x, target):
        return x.sub(target.view(-1,x.shape[1])).mul(2)


class ReLU(Module):
    def __init__(self):
        super(ReLU, self).__init__()
        self.name = 'ReLU'
        self.params = Parameter(None)
    def forward(self, input):
        self.params.input = input
        return input.clamp(min = 0)
    def backward(self, gradwrtoutput):
        return self.params.input.sign().add(1).div(2) * gradwrtoutput


class Tanh(Module):
    def __init__(self,x = None):
        super(Tanh,self).__init__()
        self.name = 'Tanh'
        self.params = Parameter(None)
    def forward(self, input):
        self.params.input = input
        return input.tanh()
    def backward(self, gradwrtoutput):
        return (1 - self.params.input.tanh().pow(2)).mul(gradwrtoutput)



class Linear(Module):
    def __init__(self,in_dim, out_dim,bias=True,dropout = 0,init='xavier'):
        self.bias = bias
        if self.bias:
            #self.params = Parameter(torch.randn((in_dim + 1,out_dim))/(math.sqrt(in_dim /2)))
            self.params = Parameter(2*(torch.rand((in_dim + 1,out_dim)) - 0.5)/(math.sqrt(in_dim )))
        else:
            #self.params = Parameter(torch.randn((in_dim,out_dim))/(math.sqrt(in_dim/2)))
            self.params = Parameter(2*(torch.rand((in_dim + 1,out_dim)) - 0.5)/(math.sqrt(in_dim )))
        if dropout > 0:
            self.dropout = dropout
        else:
            self.dropout = None
        
    
    def forward(self, x):
        if self.dropout is not None:
            self.non_zero = torch.bernoulli(self.dropout * torch.ones((self.params.data.size(1),)))
        else:
            self.non_zero = torch.ones((self.params.data.size(1),))
            
        if self.bias:
            self.params.input = torch.cat((x,torch.ones((x.size(0),1))),dim=1)    
        else:
            self.params.input = x
        return  (self.params.input @self.params.data)* self.non_zero
    
    
    def backward(self, gradwrtoutput): 
        self.params.grad = torch.bmm((gradwrtoutput*self.non_zero).unsqueeze(-1) , self.params.input.unsqueeze(-2)).mean(axis=0).t()
        if self.bias:
            return ((gradwrtoutput*self.non_zero) @ self.params.data.t())[:,:-1]
        else:
            return (gradwrtoutput*self.non_zero) @ self.params.data.t()
    


class Sequential(Module):
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.modules = modules
        self.params = []
        for mod in self.modules:
            if mod.param() is not None:
                self.params.append(mod.param())


    def forward(self, input):
        for mod in self.modules:
            input = mod.forward(input)
        return input

    def backward(self, gradwrtoutput):
        for mod in reversed(self.modules):
            gradwrtoutput = mod.backward(gradwrtoutput)





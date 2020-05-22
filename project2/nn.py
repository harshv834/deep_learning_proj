import math
import torch


class Module ( object ) :
    #Basic module class to define layers
    def forward ( self , * input ) :
        raise NotImplementedError
    def backward ( self , * gradwrtoutput ) :
        raise NotImplementedError
    def param ( self ) :
        return self.params



class Parameter(object):
    #Parameter class to store parameters, input to parameters and gradients for the parameter
    def __init__(self, data, grad=0, x=None ):
        super(Parameter, self).__init__()
        self.data = data
        self.grad = grad
        self.input = x


class lossMSE(Module):
    #Mean Squared Error Loss function
    def __init__(self):
        super(lossMSE, self).__init__()
        self.name = 'MSE_loss'

    def forward(self, x, target):
        #Forward function computes loss between predictions and target
        return x.sub(target).pow(2).mean()
    def backward(self, x, target):
        #Backward function computes gradient of loss between predictions and target
        return x.sub(target.view(-1,x.shape[1])).mul(2)


class ReLU(Module):
    #Activation function ReLU inherited from Module class
    def __init__(self):
        super(ReLU, self).__init__()
        self.name = 'ReLU'
        #For all functional layers, the parameter has value None
        self.params = Parameter(None)
    def forward(self, input):
        #Apply activation function and store the input
        self.params.input = input
        return input.clamp(min = 0)
    def backward(self, gradwrtoutput):
        #Compute gradient wrt the activation function
        return self.params.input.sign().add(1).div(2) * gradwrtoutput


class Tanh(Module):
    #Tanh activation function layer inherited from module
    def __init__(self,x = None):
        super(Tanh,self).__init__()
        self.name = 'Tanh'
        #For all functional layers, the parameter has value None
        self.params = Parameter(None)
    def forward(self, input):
        #Apply tanh on input and store the input
        self.params.input = input
        return input.tanh()
    def backward(self, gradwrtoutput):
        #Compute gradient of tanh function
        return (1 - self.params.input.tanh().pow(2)).mul(gradwrtoutput)



class Linear(Module):
    #Linear Layer inherited from Module class
    def __init__(self,in_dim, out_dim,bias=True,dropout = 0):
        self.bias = bias
        if self.bias:
            #kaiming initialization of weights, bias included as additional dimension in the weights
            self.params = Parameter(2*(torch.rand((in_dim + 1,out_dim)) - 0.5)/(math.sqrt(in_dim )))
        else:
            self.params = Parameter(2*(torch.rand((in_dim + 1,out_dim)) - 0.5)/(math.sqrt(in_dim )))
        if dropout > 0:
            #Dropout indicator stored
            self.dropout = dropout
        else:
            self.dropout = None
        
    
    def forward(self, x):

        if self.dropout is not None:
            #Compute nodes to keep with dropout
            self.non_zero = torch.bernoulli((1 - self.dropout) * torch.ones((self.params.data.size(1),)))
        else:
            #If no dropout, keep all nodes
            self.non_zero = torch.ones((self.params.data.size(1),))
            

        #Store input
        if self.bias:
            self.params.input = torch.cat((x,torch.ones((x.size(0),1))),dim=1)    
        else:
            self.params.input = x
        #Compute output as matrix multiplication
        return  (self.params.input @self.params.data)* self.non_zero
    
    
    def backward(self, gradwrtoutput): 
        #Compute gradient wrt weights using error from previous layer
        self.params.grad = torch.bmm((gradwrtoutput*self.non_zero).unsqueeze(-1) , self.params.input.unsqueeze(-2)).mean(axis=0).t()
        #Return error for current layer
        if self.bias:
            return ((gradwrtoutput*self.non_zero) @ self.params.data.t())[:,:-1]
        else:
            return (gradwrtoutput*self.non_zero) @ self.params.data.t()
    


class Sequential(Module):
    #Sequential container to create model using list of layers
    def __init__(self, *modules):
        super(Sequential, self).__init__()
        self.modules = modules
        self.params = []
        #Parameters of complete model is a list of parameters of the layers
        for mod in self.modules:
            if mod.param() is not None:
                self.params.append(mod.param())


    def forward(self, input):
        #Forward function for container applies input to each layer and uses its output as the input for the next layer
        for mod in self.modules:
            input = mod.forward(input)
        return input

    def backward(self, gradwrtoutput):
        #Bacward function for container uses gradient wrt output to compute gradients for each layer in the reverse order
        for mod in reversed(self.modules):
            gradwrtoutput = mod.backward(gradwrtoutput)





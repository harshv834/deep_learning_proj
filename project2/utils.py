import torch
import math


def generate_disc_set(nb):
    #Function to generate data and labels
    #generate data uniformly from unit square
    data = torch.empty(nb,2).uniform_(0,1)
    #Generate labels as -1 or 1 if they lie inside or outside the cirlce centered at (0.5,0.5) with radius (1/sqrt(2 pi))
    target = (data-0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign()


    return data, target


def compute_nb_acc(model, data_input, data_target, mini_batch_size):
    # Function that returns accuracy
    nb_acc = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        #Forward pass over the model in mini-batches
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        predicted_classes,_ = output.max(1)
        #Compute accuracy for each minibatch
        nb_acc = nb_acc + (data_target.narrow(0, b, mini_batch_size).eq(predicted_classes.sign())).sum()
    return nb_acc*100.0/data_input.size()[0]





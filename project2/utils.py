import torch
import math

def generate_disc_set(nb):
    data = torch.empty(nb,2).uniform_(0,1)
    target = (data-0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign()
    #.add(1).div(2)

    return data, target


def convert_to_one_hot_labels(target):
    hot_labels = empty(target.size(0), 2)
    hot_labels[:,0], hot_labels[:,1] = 1-target, target 
    return hot_labels

def compute_nb_acc(model, data_input, data_target, mini_batch_size):
    # Function that returns the number of errors
    nb_acc = 0
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        predicted_classes,_ = output.max(1)
        nb_acc = nb_acc + (data_target.narrow(0, b, mini_batch_size).eq(predicted_classes.sign())).sum()
    return nb_acc*100.0/data_input.size()[0]





import torch
torch.set_grad_enabled(False)
import tqdm
import argparse
from optimizers import *
from nn import *
from utils import *
import matplotlib.pyplot as plt
import pickle 
#Import all modules and classes for the model




#Define model as 3 layers with 25 hidden units each, 1 output node and activation functions ReLU, ReLU, Tanh, Tanh
model = Sequential(Linear(2, 25), ReLU(),Linear(25, 25), ReLU(),Linear(25, 25,), Tanh(),Linear(25, 1), Tanh())

# Setting the training parameter
mini_batch_size, epochs, lr = 1 , 100, 1e-4
#Define Loss
criterion = lossMSE()
#Initialize optimizer, here SGD without momentum
optimi = SGD_opti(model.param(), learn_rate=lr,beta = 0.0)

# Generating the train and test datasets
nb_samples = 1000

train_input, train_target = generate_disc_set(nb_samples)
test_input, test_target = generate_disc_set(nb_samples)
#Normalize training and test samples
mu, std_dev = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std_dev)
test_input.sub_(mu).div_(std_dev)


#Create lists to store loss, training and test accuracy during epochs
loss_list = []
train_acc_list = []
test_acc_list = []

for e in tqdm.tqdm(range(0, epochs)):
    sum_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
        #Forget previous gradients
        optimi.zero_grad()
        #Compute model output
        output = model.forward(train_input.narrow(0, b, mini_batch_size))
        #Compute loss for the mini-batch
        loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
        sum_loss+= loss
        #Compute gradient wrt loss
        first_grad = criterion.backward(output, train_target.narrow(0, b, mini_batch_size))
        #Apply backward pass with gradient wrt loss
        model.backward(first_grad)
        #Update model weights using optimizer
        optimi.step()
        
    loss_list.append(sum_loss.item()/train_input.size(0))
    #Compute training and test accuracies with current model
    train_acc = compute_nb_acc(model, train_input, train_target, 100)
    test_acc = compute_nb_acc(model, test_input, test_target, 100)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    #Print current model accuracies and loss
    if e%20 ==19:
        print("Epoch %d : Loss - %.7f , Training accuracy - %.2f , Test accuracy - %.2f "%(e,sum_loss/nb_samples,train_acc,test_acc))


#Plot loss                
plt.plot(range(len(loss_list)),loss_list)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.show()

#Plot training and test accuracies

plt.plot(range(len(train_acc_list)), train_acc_list,label="train_acc")
plt.plot(range(len(test_acc_list)),test_acc_list,label="test_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

#Print final training and test accuracies
print("Final Training accuracy - %.2f ,Final Test accuracy - %.2f "%(train_acc_list[-1],test_acc_list[-1]))

torch.save(model,open("model.pickle","wb"))

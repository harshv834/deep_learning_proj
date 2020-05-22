import torch
torch.set_grad_enabled(False)
import tqdm
import argparse
from optimizers import *
from nn import *
from utils import *
import matplotlib.pyplot as plt






model = Sequential(Linear(2, 25), ReLU(),Linear(25, 25), ReLU(),Linear(25, 25,), Tanh(),Linear(25, 1), Tanh())

# Setting the training parameter
mini_batch_size, epochs, lr = 1 , 200, 1e-4
criterion = lossMSE()
optimi = SGD_opti(model.param(), learn_rate=lr,beta=0.9)
scheduler = LR_scheduler(optimi,counter = epochs/4, gamma = 0.5)


# Generating the train and test datasets
nb_samples = 1000

train_input, train_target = generate_disc_set(nb_samples)


test_input, test_target = generate_disc_set(nb_samples)
mu, std_dev = train_input.mean(), train_input.std()
train_input.sub_(mu).div_(std_dev)
test_input.sub_(mu).div_(std_dev)


loss_list = []
train_acc_list = []
test_acc_list = []
for e in tqdm.tqdm(range(0, epochs)):
    sum_loss = 0
    for b in range(0, train_input.size(0), mini_batch_size):
        optimi.zero_grad()
        output = model.forward(train_input.narrow(0, b, mini_batch_size))
        loss = criterion.forward(output, train_target.narrow(0, b, mini_batch_size))
        sum_loss+= loss
        first_grad = criterion.backward(output, train_target.narrow(0, b, mini_batch_size))
        model.backward(first_grad)
        optimi.step()
    #scheduler.step()
    loss_list.append(sum_loss.item()/train_input.size(0))
    train_acc = compute_nb_acc(model, train_input, train_target, mini_batch_size)
    test_acc = compute_nb_acc(model, test_input, test_target,mini_batch_size)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)


plt.plot(range(len(loss_list)),loss_list)
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
plt.savefig("loss_mom.jpg")
plt.show()



plt.plot(range(len(train_acc_list)), train_acc_list,label="train_acc")
plt.plot(range(len(test_acc_list)),test_acc_list,label="test_acc")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy_mom.jpg")
plt.show()

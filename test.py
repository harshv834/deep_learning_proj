import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import dlc_practical_prologue as prologue
#%matplotlib inline
N=1000
from torch.utils.data import DataLoader, Dataset
import tqdm
from torch.autograd import Variable
from models import CompareNet1,CompareNet11,CompareNet12,CompareNet2,CompareNet21,CompareNet22
# Chosen best architecture:
# 1. Weight Sharing
# 2. Without auxillary loss
# 3. Batch normalisation
# 4. Activation function: ReLU
# 5. Convolution layers for the base
print(
    'Chosen best architecture has convolution layers for feature extraction, weight sharing, no auxillary loss with Batch normalization:CompareNet22')
print(
    'We also observed that an architecture with convolution layers for feature extraction, auxillary loss, Batch normalization with no weight sharing:CompareNet21, also gives similar results')




class DigitPairsDataset(Dataset):
    def __init__(self, img_pair, targets, classes):
        super(DigitPairsDataset, self).__init__()
        self.img_pair = img_pair
        self.targets = targets
        self.classes = classes

    def __len__(self):
        return self.targets.size()[0]

    def __getitem__(self, idx):
        return self.img_pair[idx], self.targets[idx], self.classes[idx]


criterion = nn.CrossEntropyLoss()
auxillary = False  # choose to have auxillary loss or not
linear = False  # choose to have linear layers or not for base
wt_sharing = True  # choose to have weight sharing or not
conv = True  # choose to have convolution layers for base or not
batch_normalization = True  # choose to have batch normalisation for base or not

if not wt_sharing and not auxillary:
    print('Please enable one of weight sharing or auxillary')

if batch_normalization and linear:
    print('Batch normalization is only applied with convolution layers')

# chosing net based on user input
if linear and auxillary and wt_sharing:
    net = CompareNet1()
    flag = 1
elif linear and auxillary and not wt_sharing:
    net = CompareNet11()
    flag = 11
elif linear and not auxillary and wt_sharing:
    net = CompareNet12()
    flag = 12
elif conv and auxillary and wt_sharing:
    net = CompareNet2(batch_normalization=batch_normalization)
    flag = 2
elif conv and auxillary and not wt_sharing:
    net = CompareNet21(batch_normalization=batch_normalization)
    flag = 21
elif conv and not auxillary and wt_sharing:
    net = CompareNet22(batch_normalization=batch_normalization)
    flag = 22

nb_iterations = 1
count = 0
epochs = 50
print(
    'This code generates runs the best chosen solution for {nb_iterations} iteration. To increase the number of iterations, please change nb_iterations'.format(
        nb_iterations=nb_iterations))
mu = 1.0  # parameter to control weightage given to auxillary loss
optimizer = optim.Adam(net.parameters(), lr=0.001)


# function to calculate accuracy

def calc_accuracy(data_loader, model):
    correct_count = 0.0
    for i, data in enumerate(data_loader, 0):
        img_pair, target, classes = data
        if auxillary and wt_sharing:
            pred_sign, pred_class0, pred_class1 = net(img_pair)
            loss = criterion(pred_sign, target) + mu * (
                        criterion(pred_class0, classes[:, 0]) + criterion(pred_class1, classes[:, 1]))
        elif auxillary and not wt_sharing:
            pred_sign, pred_class = net(img_pair)  # to remove weight sharing
            classes = 10 * classes[:, 1] + classes[:, 0]
            loss = criterion(pred_sign, target) + mu * criterion(pred_class, classes)
        elif not auxillary and wt_sharing:
            pred_sign = net(img_pair)  # for removing weight sharing
            loss = criterion(pred_sign, target)
        pred = torch.argmax(pred_sign, -1)
        correct_count += int((target.eq(pred)).sum())
    return correct_count * 100.0 / N


while count < nb_iterations:
    loss_arr = []
    train_acc_arr = []
    val_acc_arr = []
    train_input, train_target, train_classes, test_input, test_target, test_classes = prologue.generate_pair_sets(N)
    train_dataset = DigitPairsDataset(train_input, train_target, train_classes)
    test_dataset = DigitPairsDataset(test_input, test_target, test_classes)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)
    for epoch in tqdm.tqdm(range(epochs)):
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            img_pair, target, classes = data

            optimizer.zero_grad()

            if auxillary and wt_sharing:
                pred_sign, pred_class0, pred_class1 = net(img_pair)
                loss = criterion(pred_sign, target) + mu * (
                            criterion(pred_class0, classes[:, 0]) + criterion(pred_class1, classes[:, 1]))

            elif auxillary and not wt_sharing:
                pred_sign, pred_class = net(img_pair)
                classes = 10 * classes[:, 1] + classes[:, 0]
                loss = criterion(pred_sign, target) + mu * criterion(pred_class, classes)

            elif not auxillary and wt_sharing:
                pred_sign = net(img_pair)
                loss = criterion(pred_sign, target)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        net.eval()
        running_loss /= N
        loss_arr.append(running_loss)
        train_acc = calc_accuracy(train_loader, net)
        val_acc = calc_accuracy(test_loader, net)
        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)
        if epoch % 10 == 9:
            print("Epoch : %d  ,   Train Accuracy : %.2f  , Validation Accuracy : %.2f , Training Loss : %.6f" % (
            epoch, train_acc, val_acc, running_loss))

    # plotting loss curve
    plt.plot(range(epochs), loss_arr)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss over iterations")
    fileplot1 = "plot_loss_{count}.png".format(count=count)
    plt.savefig(fileplot1)
    plt.show()

    # plotting accuracy curves
    plt.plot(range(epochs), train_acc_arr, label=" Training accuracy")
    plt.plot(range(epochs), val_acc_arr, label="Validation accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.title("Training and Validation accuracy during training")
    plt.legend()
    fileplot = "plot_acc_{count}.png".format(count=count)
    plt.savefig(fileplot)
    plt.show()

    fileloss = "loss_{count}.npy".format(count=count)
    np.save(fileloss, loss_arr)
    filetrain = "train_acc_{count}.npy".format(count=count)
    np.save(filetrain, train_acc_arr)
    fileval = "val_acc_{count}.npy".format(count=count)
    np.save(fileval, val_acc_arr)
    count += 1

import numpy as np
import matplotlib.pyplot as plt

# this file works with npy files created using the test.py file with multiple iterations
# insert the name of the folder for the "case" variable
case = 'conv_bn_wt'
acc_train = []
acc_val = []
nb_iterations = 11
for i in range(nb_iterations):
    filename_tr = "{case}/train_acc_{i}.npy".format(case=case, i=i)
    acc_train.append(np.load(filename_tr))
    filename_vl = "{case}/val_acc_{i}.npy".format(case=case, i=i)
    acc_val.append(np.load(filename_vl))

acc_train = np.array(acc_train)
st_dev_train = np.std(acc_train, 1)
avg_train = np.mean(st_dev_train)

acc_val = np.array(acc_val)
st_dev_val = np.std(acc_val, 1)
avg_val = np.mean(st_dev_val)

plt.plot(range(nb_iterations), st_dev_train, label="Standard Deviation per iteration")
plt.plot(range(nb_iterations), avg_train * np.ones(nb_iterations), label="Average Standard Deviation")
plt.ylabel("Standard Deviation")
plt.xlabel("Iterations")
plt.legend()
plt.title("Standard deviation for Training accuracy")
fileplot = "{case}/plot_st_dev_train_{case}.png".format(case=case)
plt.savefig(fileplot)
plt.show()

plt.plot(range(nb_iterations), st_dev_val, label="Standard Deviation per iteration")
plt.plot(range(nb_iterations), avg_val * np.ones(nb_iterations), label="Average Standard Deviation")
plt.ylabel("Standard Deviation")
plt.xlabel("Iterations")
plt.legend()
plt.title("Standard deviation for Validation accuracy")
fileplot = "{case}/plot_st_dev_val_{case}.png".format(case=case)
plt.savefig(fileplot)
plt.show()
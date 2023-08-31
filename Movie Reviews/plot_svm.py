import numpy as np
import matplotlib.pyplot as plt

data_L = np.loadtxt('results_svm.txt')
iter = [1000, 2500, 5000, 7500]

for K in iter:

    data = data_L[:, data_L[0]==K]

    learning_rates = data[1,:]
    regularization_params = data[2,:]
    training_accs = data[3,:]
    validation_accs = data[4,:]

    # create a meshgrid for the learning rate and regularization parameter values
    lr_mesh, rp_mesh = np.meshgrid(np.unique(learning_rates), np.unique(regularization_params))

    # reshape the training and validation accuracy values into the same shape as the meshgrid
    training_accs_mesh = training_accs.reshape(lr_mesh.shape)
    validation_accs_mesh = validation_accs.reshape(lr_mesh.shape)

    # create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))

    for i in range(lr_mesh.shape[0]):
        for j in range(lr_mesh.shape[1]):
            text = ax1.text(j, i, "{:.2f}".format(training_accs_mesh[i, j]*100), ha="center", va="center", color="black")

    # plot the training accuracy in the left subplot
    im1 = ax1.imshow(training_accs_mesh, cmap='RdYlGn')
    ax1.set_yticks(np.arange(len(np.unique(learning_rates))))
    ax1.set_yticklabels(np.unique(learning_rates))
    ax1.set_xticks(np.arange(len(np.unique(regularization_params))))
    ax1.set_xticklabels(np.unique(regularization_params))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    ax1.set_title("Training Accuracy Grid Search")
    ax1.set_ylabel("Learning Rate")
    ax1.set_xlabel("Regularization Parameter")

    for i in range(rp_mesh.shape[0]):
        for j in range(rp_mesh.shape[1]):
            text = ax2.text(j, i, "{:.2f}".format(validation_accs_mesh[i, j]*100), ha="center", va="center", color="black")

    # plot the validation accuracy in the right subplot
    im2 = ax2.imshow(validation_accs_mesh, cmap='RdYlGn')
    ax2.set_yticks(np.arange(len(np.unique(learning_rates))))
    ax2.set_yticklabels(np.unique(learning_rates))
    ax2.set_xticks(np.arange(len(np.unique(regularization_params))))
    ax2.set_xticklabels(np.unique(regularization_params))
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    ax2.set_title("Validation Accuracy Grid Search")
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Regularization Parameter")

    # adjust the layout of the subplots to avoid overlap of the y-axis labels
    fig.suptitle('SVM with '+str(K)+' iterations')
    fig.tight_layout()
    # show the plot
    plt.show()
    #fig.savefig('img/svm'+str(K)+'.png')
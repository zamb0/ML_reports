import numpy as np
import matplotlib.pyplot as plt
import os
import pvml

def read_acc_file(file):
    acc = np.loadtxt(file)
    train = acc[0, :]
    test = acc[1, :]

    #print(train)
    #print(test)

    plt.plot([i for i in range(30)], train, label='Train')
    plt.plot([i for i in range(30)], test, label='Test')
    plt.title('Fine Tuning')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.03)
    plt.grid(True)

    plt.savefig('img/cnn_fine.png', dpi=300)
    plt.show()

def read_acc_dir(dir):
    plt.figure(figsize=(15, 10))
    for file in os.listdir(dir):

        if file.endswith('.txt'):
            acc = np.loadtxt(dir+'/'+file)
            train = acc[0, :]
            test = acc[1, :]

            #print(train)
            #print(test)

            plt.subplot(2, 1, 1)
            plt.plot([i for i in range(5000) if i%100 == 0], train, label=file[4:-4])
            plt.title('Train')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.ylim(0, 0.3)
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot([i for i in range(5000) if i%100 == 0], test, label=file[4:-4])
            plt.title('Test')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.ylim(0, 0.3)
            plt.grid(True)

    plt.tight_layout()
    plt.savefig('img/mlp_data.png', dpi=300)
    plt.show()
    

read_acc_dir('mlp_res')
read_acc_file('cnn_res/acc_fine_tuning.txt')



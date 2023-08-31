import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(8, 6), dpi=500)

data = np.loadtxt('results.txt')

plt.imshow(data[1:], cmap='RdYlGn', interpolation='nearest')
plt.yticks(np.arange(0, 2), ['Train', 'Validation'])
plt.xticks(np.arange(0, len(data[0])), data[0].astype(int))
plt.xlabel('Vocabulary size')

#add text to the plot

for i in range(2):
    for j in range(len(data[0])):
        plt.text(j, i, round(data[i+1, j]*100, 2), horizontalalignment='center', verticalalignment='center')
    
#plt.savefig('img/acc.png', dpi=500)
#plt.savefig('img/acc_stop.png', dpi=500)
#plt.savefig('img/acc_stop&stem.png', dpi=500)




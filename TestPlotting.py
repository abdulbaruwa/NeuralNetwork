import matplotlib.pyplot as plt
import numpy as np

epoch = [1, 2, 3, 4, 5, 6, 7]
perf = [95.24, 96.05, 96.02, 95.66, 95.68, 96.02, 95.31]

plt.plot(epoch, perf)

plt.xlabel('Number of epochs')
plt.ylabel('Performance')
plt.title('Perf and Epoch on MNIST dataset for 3-Layer N-Net')
plt.grid()
plt.savefig("FirstNet.png")
plt.show()

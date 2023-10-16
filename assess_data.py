import numpy as np
import matplotlib.pyplot as plt

# Load the data
test_data = np.loadtxt('signal_data/test_data.txt')
train_data = np.loadtxt('signal_data/train_data.txt')
test_labels = np.loadtxt('signal_data/test_labels.txt')
train_labels = np.loadtxt('signal_data/train_labels.txt')

print(len(train_data), len(test_data))
# Plot the data
freq = np.arange(6, 55, 0.1)
[plt.plot(freq, test_labels[i], label='train') for i in range(len(test_labels))]
plt.show()
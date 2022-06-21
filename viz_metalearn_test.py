import numpy as np
import matplotlib.pyplot as plt
from neural_nets.backprop import BackPropNet
import sys
import os

dir = sys.argv[1]

errorhistE = np.load(os.path.join(dir,"test_evolved.npy"))
errorhistR = np.load(os.path.join(dir,"test_random.npy"))

plt.plot(np.mean(errorhistE,axis=1).T,'r',alpha=0.1)
plt.plot(np.mean(errorhistR,axis=1).T,'y',alpha=0.1)
plt.plot(np.mean(np.mean(errorhistE,axis=1),axis=0),'k-')
plt.plot(np.mean(np.mean(errorhistR,axis=1),axis=0),'k--')

plt.xlabel("Learning epochs")
plt.ylabel("Error")
plt.show()

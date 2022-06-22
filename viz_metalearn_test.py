import numpy as np
import matplotlib.pyplot as plt
from neural_nets.backprop import BackPropNet
import sys
import os

# dir = sys.argv[1]
#
# errorhistE = np.load(os.path.join(dir,"test_evolved.npy"))
# errorhistR = np.load(os.path.join(dir,"test_random.npy"))
#
# plt.plot(np.mean(errorhistE,axis=1).T,'r',alpha=0.1)
# plt.plot(np.mean(errorhistR,axis=1).T,'y',alpha=0.1)
# plt.plot(np.mean(np.mean(errorhistE,axis=1),axis=0),'k-')
# plt.plot(np.mean(np.mean(errorhistR,axis=1),axis=0),'k--')
#
# plt.xlabel("Learning epochs")
# plt.ylabel("Error")
# plt.show()

E2 = np.load("ML2/test_evolved.npy")
R2 = np.load("ML2/test_random.npy")
E4 = np.load("ML4/test_evolved.npy")
R4 = np.load("ML4/test_random.npy")
E8 = np.load("ML8/test_evolved.npy")
R8 = np.load("ML8/test_random.npy")

plt.plot(np.mean(np.mean(E2,axis=1),axis=0),'r-')
plt.plot(np.mean(np.mean(E4,axis=1),axis=0),'g-')
plt.plot(np.mean(np.mean(E8,axis=1),axis=0),'b-')

plt.plot(np.mean(np.mean(R2,axis=1),axis=0),'y-')
plt.plot(np.mean(np.mean(R4,axis=1),axis=0),'y-')
plt.plot(np.mean(np.mean(R8,axis=1),axis=0),'y-')

plt.xlabel("Learning epochs")
plt.ylabel("Error")
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from neural_nets.backprop import BackPropNet
import sys
import os

F = int(sys.argv[1])
T = int(sys.argv[2])

# Save directory
dir = 'Learning'

# Problem dataset
K = 25
amplitude = 0.9
phase = np.pi #0.0
inputs = np.linspace(-5,5,K).reshape(K,1)
outputs = amplitude*np.sin(inputs+phase)

# Load
learnHistory = []
genotype = []
for E in range(F,T+1):
    learnHistory.append(np.load(os.path.join(dir, "best_{}.npy".format(E))))
    genotype.append(np.load(os.path.join(dir, "gen_{}.npy".format(E))))

# Show learning
for E in range(F,T+1):
    plt.plot(learnHistory[E])
plt.plot(np.mean(learnHistory,axis=0),'k')
plt.xlabel("Epochs")
plt.ylabel("Fitness")
plt.show()

# Show run
plt.plot(inputs,outputs,'k')
for E in range(F,T+1):
    unitsperlayer = [1, 5, 5, 1]
    genesize = np.sum([unitsperlayer[j]*unitsperlayer[j+1] for j in range(len(unitsperlayer)-1)])
    nn = BackPropNet(unitsperlayer, activation="tanh")
    nn.setParams(genotype[E])
    plt.plot(inputs,nn.forward(inputs)[-1],'o-')
plt.show()

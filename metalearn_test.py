import numpy as np
import matplotlib.pyplot as plt
from neural_nets.backprop import BackPropNet
import sys
import os

FROM = int(sys.argv[1])
TO = int(sys.argv[2])
dir = sys.argv[3]

# Create dataset with  T different tasks and K different points per task
L = 100 # Number of epochs to train
K = 25  # Training -- Number of different points to take from the sine wave
KK = 100 # Testing
train_samples_per_epoch = 25 # Number of different samples to take

# Load
genotype = []
for E in range(FROM,TO+1):
    genotype.append(np.load(os.path.join(dir, "gen_{}.npy".format(E))))

# Genotype to phenotype mapping
unitsperlayer = [1, 10, 10, 1]
genesize = np.sum([(unitsperlayer[j]+1)*unitsperlayer[j+1] for j in range(len(unitsperlayer)-1)])

# Test in an unseen task
T = 100        # Number of different tasks (sine waves)
amplitude =  np.random.random(size=T)*0.9 #[0.9]*T
phase = np.random.random(size=T)*2*np.pi

inputs_train = np.linspace(-2*np.pi,2*np.pi,K).reshape(K,1)
outputs_train = np.zeros((T,K,1))
for i in range(T):
    outputs_train[i] = amplitude[i]*np.sin(inputs_train+phase[i])

inputs_test = np.linspace(-2*np.pi,2*np.pi,KK).reshape(KK,1)
outputs_test = np.zeros((T,KK,1))
for i in range(T):
    outputs_test[i] = amplitude[i]*np.sin(inputs_test+phase[i])

# Test and show the evolved ones first
errorhistE = np.zeros((TO+1-FROM,T,L))
for E in range(FROM,TO+1):
    genotype = np.load(os.path.join(dir, "gen_{}.npy".format(E)))
    for i in range(T):
        nn = BackPropNet(unitsperlayer, activation="tanh")
        params = genotype.copy()
        nn.setParams(params)
        for j in range(L):
            errorhistE[E,i,j] = nn.training_step(inputs_train, outputs_train[i])[0]

# Test and show
errorhistR = np.zeros((TO+1-FROM,T,L))
for E in range(FROM,TO+1):
    genotype = np.random.random(genesize)* 2 - 1
    for i in range(T):
        nn = BackPropNet(unitsperlayer, activation="tanh")
        params = genotype.copy()
        nn.setParams(params)
        for j in range(L):
            errorhistR[E,i,j] = nn.training_step(inputs_train, outputs_train[i])[0]

# Save data
np.save(os.path.join(dir,"test_evolved.npy"), errorhistE)
np.save(os.path.join(dir,"test_random.npy"), errorhistR)

plt.plot(np.mean(errorhistE,axis=1).T,'r',alpha=0.2)
plt.plot(np.mean(errorhistR,axis=1).T,'y',alpha=0.2)
plt.plot(np.mean(np.mean(errorhistE,axis=1),axis=0),'r-')
plt.plot(np.mean(np.mean(errorhistR,axis=1),axis=0),'y-')
plt.xlabel("Learning epochs")
plt.ylabel("Error")
plt.show()

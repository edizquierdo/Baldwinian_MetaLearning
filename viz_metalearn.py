import numpy as np
import matplotlib.pyplot as plt
from neural_nets.backprop import BackPropNet
import sys
import os

FROM = int(sys.argv[1])
TO = int(sys.argv[2])
dir = sys.argv[3]

# Create dataset with  T different tasks and K different points per task
L = 10 # Number of epochs to train
K = 25  # Training -- Number of different points to take from the sine wave
KK = 100 # Testing
train_samples_per_epoch = 25 # Number of different samples to take

T = 4       # Number of different tasks (sine waves)
amplitude = [0.9]*T #np.random.random(size=T)*0.9 + 0.1
#phase = np.linspace(0.0, np.pi, T) # np.random.random(size=T)* np.pi     #[0.0,Pi]
phase = np.linspace(0.0, 3*np.pi/2, T) # np.random.random(size=T)* np.pi     #[0.0,Pi]

inputs_train = np.linspace(-5,5,K).reshape(K,1)
outputs_train = np.zeros((T,K,1))
for i in range(T):
    outputs_train[i] = amplitude[i]*np.sin(inputs_train+phase[i])

inputs_test = np.linspace(-5,5,KK).reshape(KK,1)
outputs_test = np.zeros((T,KK,1))
for i in range(T):
    outputs_test[i] = amplitude[i]*np.sin(inputs_test+phase[i])

# Load
bestHistory = []
avgHistory = []
genotype = []
for E in range(FROM,TO+1):
    bestHistory.append(np.load(os.path.join(dir, "best_{}.npy".format(E))))
    genotype.append(np.load(os.path.join(dir, "gen_{}.npy".format(E))))

# Show evolution
for E in range(FROM,TO+1):
    plt.plot(-bestHistory[E])
plt.plot(-np.mean(bestHistory,axis=0),'k')
plt.xlabel("Generations")
plt.ylabel("Error")
plt.show()

# Genotype to phenotype mapping
unitsperlayer = [1, 10, 10, 1]
genesize = np.sum([(unitsperlayer[j]+1)*unitsperlayer[j+1] for j in range(len(unitsperlayer)-1)])

# Test and show
for E in range(FROM,TO+1):
    genotype = np.load(os.path.join(dir, "gen_{}.npy".format(E)))
    for i in range(T):
        nn = BackPropNet(unitsperlayer, activation="tanh")
        params = genotype.copy()
        nn.setParams(params)
        plt.plot(inputs_test,nn.forward(inputs_test)[-1],'k')
        for j in range(L):
            nn.training_step(inputs_train, outputs_train[i])[0]
        plt.plot(inputs_test,outputs_test[i])
        plt.plot(inputs_test,nn.forward(inputs_test)[-1],'o-')
plt.show()

# Test in an unseen task
T = 1       # Number of different tasks (sine waves)
amplitude = np.random.random(size=T)*0.9 + 0.1
phase = np.random.random(size=T)*2*np.pi #[3*np.pi/2] #np.linspace(np.pi/2, 3*(np.pi/2), T) # np.random.random(size=T)* np.pi     #[0.0,Pi]

inputs_train = np.linspace(-5,5,K).reshape(K,1)
outputs_train = np.zeros((T,K,1))
for i in range(T):
    outputs_train[i] = amplitude[i]*np.sin(inputs_train+phase[i])

inputs_test = np.linspace(-5,5,KK).reshape(KK,1)
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
        plt.plot(inputs_test,nn.forward(inputs_test)[-1],'k')
        for j in range(L):
            errorhistE[E,i,j] = nn.training_step(inputs_train, outputs_train[i])[0]
        plt.plot(inputs_test,outputs_test[i])
        plt.plot(inputs_test,nn.forward(inputs_test)[-1],'o-')
plt.show()

# Test and show
errorhistR = np.zeros((TO+1-FROM,T,L))
for E in range(FROM,TO+1):
    genotype = np.random.random(genesize)* 2 - 1
    for i in range(T):
        nn = BackPropNet(unitsperlayer, activation="tanh")
        params = genotype.copy()
        nn.setParams(params)
        plt.plot(inputs_test,nn.forward(inputs_test)[-1],'k')
        for j in range(L):
            errorhistR[E,i,j] = nn.training_step(inputs_train, outputs_train[i])[0]
        plt.plot(inputs_test,outputs_test[i])
        plt.plot(inputs_test,nn.forward(inputs_test)[-1],'o-')
plt.show()

plt.plot(np.mean(np.mean(errorhistE,axis=1),axis=0))
plt.plot(np.mean(np.mean(errorhistR,axis=1),axis=0))
plt.xlabel("Learning epochs")
plt.ylabel("Error")
plt.show()

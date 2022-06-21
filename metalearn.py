import numpy as np
import matplotlib.pyplot as plt
import mga
from neural_nets.backprop import BackPropNet
import sys
import os

E = int(sys.argv[1])

# Save directory
saveDir = 'MetaLearning4'
if not os.path.exists(saveDir):
    os.mkdir(saveDir)

# Create dataset with  T different tasks and K different points per task
L = 10 # Number of epochs to train
K = 25  # Number of different points to take from the sine wave

T = 4       # Number of different tasks (sine waves)
amplitude = [0.9]*T #np.random.random(size=T)*0.9 + 0.1
#phase = np.linspace(0.0, np.pi, T) # np.random.random(size=T)* np.pi     #[0.0,Pi]
phase = np.linspace(0.0, 3*np.pi/2, T) # np.random.random(size=T)* np.pi     #[0.0,Pi]

inputs = np.linspace(-5,5,K).reshape(K,1)
outputs = np.zeros((T,K,1))
for i in range(T):
    outputs[i] = amplitude[i]*np.sin(inputs+phase[i])

# Genotype to phenotype mapping
unitsperlayer = [1, 10, 10, 1]
genesize = np.sum([(unitsperlayer[j]+1)*unitsperlayer[j+1] for j in range(len(unitsperlayer)-1)])

# EA Params
popsize = 100
recombProb = 0.5
mutatProb = 0.025
demeSize = 2
generations = 1000

def fitnessFunction(genotype):
    errors = np.zeros((T,L))
    for i in range(T):
        nn = BackPropNet(unitsperlayer, activation="tanh")
        params = genotype.copy()
        nn.setParams(params)
        for j in range(L):
            errors[i,j] = nn.training_step(inputs, outputs[i])[0]
    return -np.mean(errors)

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()

# Get best evolved network and show its activity
af,bf,genotype = ga.fitStats()

# Save data
np.save(os.path.join(saveDir, "best_{}.npy".format(E)), ga.bestHistory)
np.save(os.path.join(saveDir, "gen_{}.npy".format(E)), genotype)

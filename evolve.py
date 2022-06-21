import numpy as np
import mga
from neural_nets.backprop import BackPropNet
import sys
import os

E = int(sys.argv[1])

# Save directory
saveDir = 'Evolution'
if not os.path.exists(saveDir):
    os.mkdir(saveDir)

# Problem dataset
K = 25
amplitude = 0.9
phase = 0.0
inputs = np.linspace(-5,5,K).reshape(K,1)
outputs = amplitude*np.sin(inputs+phase)

# Genotype to phenotype mapping
unitsperlayer = [1, 5, 5, 1]
genesize = np.sum([(unitsperlayer[j]+1)*unitsperlayer[j+1] for j in range(len(unitsperlayer)-1)])

# EA Params
popsize = 100
recombProb = 0.5
mutatProb = 0.025
demeSize = 2
generations = 100

# Fitness function
def fitnessFunction(genotype):
    nn = BackPropNet(unitsperlayer, activation="tanh")
    nn.setParams(genotype)
    return -np.mean(0.5 * (outputs - nn.forward(inputs)[-1]) ** 2, 0)

# Evolve and visualize fitness over generations
ga = mga.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()

# Get best evolved network
af,bf,genotype = ga.fitStats()

# Save data
np.save(os.path.join(saveDir, "best_{}.npy".format(E)), ga.bestHistory)
np.save(os.path.join(saveDir, "gen_{}.npy".format(E)), genotype)

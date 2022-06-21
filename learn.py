import numpy as np
import matplotlib.pyplot as plt
from neural_nets.backprop import BackPropNet
import sys
import os

E = int(sys.argv[1])

# Save directory
saveDir = 'Learning'
if not os.path.exists(saveDir):
    os.mkdir(saveDir)

# Problem dataset
K = 25
amplitude = 0.9
phase = np.pi #0.0
inputs = np.linspace(-5,5,K).reshape(K,1)
outputs = amplitude*np.sin(inputs+phase)

#for train_samples_per_epoch in range(1, len(inputs) + 1):
epochs = 500
train_samples_per_epoch = 25 #K

# Genotype to phenotype mapping
unitsperlayer = [1, 5, 5, 1]
gensize=np.sum([(unitsperlayer[j]+1)*unitsperlayer[j+1] for j in range(len(unitsperlayer)-1)])
genotype = np.random.normal(0,0.5,size=gensize)

#  Train
error_hist = np.zeros(epochs)
nn = BackPropNet(unitsperlayer, activation="tanh")
nn.setParams(genotype)
for t in range(epochs):
    # train_inds = np.random.randint(0, high=len(inputs), size=[train_samples_per_epoch])
    # error_hist[t] = nn.training_step(inputs[train_inds], outputs[train_inds])[0]
    error_hist[t] = nn.training_step(inputs, outputs)[0]

# Save data
np.save(os.path.join(saveDir, "best_{}.npy".format(E)), error_hist)
np.save(os.path.join(saveDir, "gen_{}.npy".format(E)), nn.getParams())

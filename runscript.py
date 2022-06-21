import os
import sys

reps = int(sys.argv[1])
threads = int(sys.argv[2])
program = sys.argv[3]

r = 0
for k in range(reps//threads):
    for l in range(threads):
        print(r,k,l)
        if (l<threads-1):
            os.system('python '+program+' '+str(r)+' &')
        else:
            os.system('python '+program+' '+str(r))
        r += 1

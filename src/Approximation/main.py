"""

@author: Nicolas Boulle

Generate the approximation experiment results of the paper
Run the following command
python3 main.py

"""

import subprocess
import os
cwd = os.path.dirname(os.path.realpath(__file__)) 

# Run the different experiments
print('### ReLU ###')
subprocess.call('python3 KdV_relu.py', shell=True, cwd=cwd)

print('### Polynomial ###')
subprocess.call('python3 KdV_poly.py', shell=True, cwd=cwd)

print('### Sinusoid ###')
subprocess.call('python3 KdV_sine.py', shell=True, cwd=cwd)

RP = [2,3,4,5]
RQ = [2,2,3,4]
for i in range(4):
    print('### Rational of type (%d, %d)' % (RP[i], RQ[i]))
    subprocess.call('python3 KdV_rat.py --rP %d --rQ %d' % (RP[i],RQ[i]), shell=True, cwd=cwd)

# Plot the results
subprocess.call('python3 plot_results.py', shell=True, cwd=cwd)
subprocess.call('python3 plot_validation.py', shell=True, cwd=cwd)

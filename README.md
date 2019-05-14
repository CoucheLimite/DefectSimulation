# A Python code for simulation of defect associated carrier dynamics in silicon

## Prerequisites:  
* [Numpy](https://www.numpy.org/)
* [Scipy](https://www.scipy.org/)
* [semiconductor](https://github.com/MK8J/semiconductor) by [MK8J](https://github.com/MK8J)

## Objectives and scopes
* It can be used to simulate single-level defects following Shockley-Read-Hall statistics or two-levels defects follwoing Sah-Shockley stastistc by numerically solving the carrier continuity equations.
* It can solve for thermal equlibrium condition, steady state illumination condition, transient condition with excitation sharply switch on or off.
* It is an 1-D simulation so no non-uniform distribution of carriers is considered.

## Here is a simple explanation of how to use it:
1. Import the defectSimu class and other necessary libraries	
```python
import numpy as np
import matplotlib.pyplot as plt
from DefectSimulation import defectSimu
```

2. Creat a defectSimu class object and define the doping, type and temperature of the silicon
```python
example = defectSimu(Ndop = 1e14,doptype = 'n',temp = 300)
```

3. Define defects and add them to the defectSimu class object, the defect type, energy level Et, capture cross sections sigma_e and sigma_h, and concentration Nt need to be defined. The defect type can be: 
* 'A' for an acceptor type sinle-level defect
* 'D' for a donor type sinle-level defect
* 'AA' for a double acceptor type two-levels defect, Et = [E--/-, E-/0]
* 'AD' for a donor acceptor type two-levels defect, Et = [E--/-, E-/0]
* 'DD' for a double donor type two-levels defect, Et = [E0/+, E+/++]  
It shoule be noted that for two-levels defects, the energy levels and capture cross sections are ordered from the transition of the most negatively charged states to the most postively charged states
```python
defect1 = {'type':'D','Et':0.,'sigma_e':1e-12,'sigma_h':1e-12,'Nt':1e12} 
defect2 = {'type':'A','Et':0.,'sigma_e':1e-12,'sigma_h':1e-12,'Nt':1e12} 
defect3 = {'type': 'DD','Nt': 1e12, 'Et': [-0.16,0.396], 'sigma_e': [1e-17, 1e-17], 'sigma_h': [1e-18, 1e-18]}
example.addDefect(defect1)
example.addDefect(defect2)
example.addDefect(defect3)
```
The current defect list can be shown with the function `PrintDefectList`
```python
example.PrintDefectList()
```
The output should look like:
```
Defect No.0: {'type': 'D', 'Et': 0.0, 'sigma_e': 1e-12, 'sigma_h': 1e-12, 'Nt': 1000000000000.0}
Defect No.1: {'type': 'A', 'Et': 0.0, 'sigma_e': 1e-12, 'sigma_h': 1e-12, 'Nt': 1000000000000.0}
Defect No.2: {'type': 'DD', 'Nt': 1000000000000.0, 'Et': [-0.16, 0.396], 'sigma_e': [1e-17, 1e-17], 'sigma_h': [1e-18, 1e-18]}
```
You can also delete one or more of the defects using `DelDefect` specify the defect index
```python
example.DelDefect(0)  ## This delete the defect with index 0 (so the defect 1)
example.PrintDefectList()
```
The output should like:
```
Defect No.0: {'type': 'A', 'Et': 0.0, 'sigma_e': 1e-12, 'sigma_h': 1e-12, 'Nt': 1000000000000.0}
Defect No.1: {'type': 'DD', 'Nt': 1000000000000.0, 'Et': [-0.16, 0.396], 'sigma_e': [1e-17, 1e-17], 'sigma_h': [1e-18, 1e-18]}
```
4. The thermal euilibrium solution should be automated updated whenever there is a change of the defect list, the electron concentration `n0`, the hole concentration `p0` and fraction of defect in each charge states `f0list` are solved
```python
print(example.n0, example.p0, example.f0list, sep='\n')
```
The output should be:
```
9.89962077369e+13
947428.977266
[[  9.99902181e-01   9.78186283e-05   0.00000000e+00]
 [  9.99911590e-01   2.00697999e-07   8.82097434e-05]]
```
The f0list is a numpy array with the dimension (no. of defect, 3). For each defect, the fraction of each charge states are listed in the 3 element array. Again the order is from the most negatively charged states to the most postively charged states. So:  
* 'A' f0list = [f-, f0, 0], the 0 here is meaningless, just to make the array length the same as two level defect
* 'D' f0list = [f-, f0, 0], the 0 here is meaningless, just to make the array length the same as two level defect
* 'AA' f0list = [f-, f0, f+]
* 'AD' f0list = [f--, f-, f0]
* 'DD' f0list = [f0, f+, f++]
5. Solve for steady state condition with `SolveSS`, the excess minority carrier density `nxc` needs to be defined, it can be one value or a list or an array of value
```python
nlist, plist, flist = example.SolveSS(nxc=np.logspace(12,16,50))
```
The outputs are: the list of electron concentration, the list of hole concentration and the list of trap charge states distribution. The list of trap charge states distribution is an array with the dimension (no. of nxc, no. of defect, 3), it is similar to the `f0list` described above.  
A simple function `calculateSSlifetime` can be used to calculate the steady state lifetime for minority carriers, majority carriers and apparent carriers. The photoconductance is also calculated (Actually the photoconductance times the thickness of sample).
```python
dminorlist, tauminorlist, dmajlist, taumajorlist, dapplist, tauapplist, condlist = example.calculateSSlifetime(nlist, plist, flist)
plt.figure('Lifetime')
plt.plot(dminorlist,tauminorlist,label='Minority')
plt.plot(dmajlist,taumajorlist,label='Majority')
plt.plot(dapplist,tauapplist,label='Apparent')
plt.legend()
plt.loglog()
plt.xlabel('Excess carrier density [cm-3]')
plt.ylabel('Lifetime [s]')
```
You should get something like this:  
![Steady state Lifetime](/example/Lifetime.png)





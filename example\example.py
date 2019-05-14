import numpy as np
import matplotlib.pyplot as plt
from DefectSimulation import defectSimu

example = defectSimu(Ndop = 1e14,doptype = 'n',temp = 300)

defect1 = {'type':'D','Et':0.,'sigma_e':1e-12,'sigma_h':1e-12,'Nt':1e12}
defect2 = {'type':'A','Et':0.,'sigma_e':1e-12,'sigma_h':1e-12,'Nt':1e12}
defect3 = {'type': 'DD','Nt': 1e12, 'Et': [-0.16,0.396], 'sigma_e': [1e-17, 1e-17], 'sigma_h': [1e-18, 1e-18]}

example.addDefect(defect1)
example.addDefect(defect2)
example.addDefect(defect3)
example.PrintDefectList()

example.DelDefect(0)  ## This delete the defect with index 0 (so the defect 1)
example.PrintDefectList()

print(example.n0, example.p0, example.f0list, sep='\n')

nlist, plist, flist = example.SolveSS(nxc=np.logspace(12,16,50))
dminorlist, tauminorlist, dmajlist, taumajorlist, dapplist, tauapplist, condlist = example.calculateSSlifetime(nlist, plist, flist)

plt.figure('Lifetime')
plt.plot(dminorlist,tauminorlist,label='Minority')
plt.plot(dmajlist,taumajorlist,label='Majority')
plt.plot(dapplist,tauapplist,label='Apparent')
plt.legend()
plt.loglog()
plt.xlabel('Excess carrier density [cm-3]')
plt.ylabel('Lifetime [s]')

plt.show()

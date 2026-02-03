import numpy as np
import ionsim

print(dir(ionsim))

r = np.array([[0.,0,0],
              [0,0,2]], order='F')
charge = np.ones(2)
# print(charge.shape)
# charge.reshape(-1, 1)
# print(charge.shape)
print(ionsim.calculate_coulombpotential(1, r, charge))
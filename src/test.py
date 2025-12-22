import numpy as np
import ionsim

print(dir(ionsim))

r = np.array([[0.,0,0],
              [0,0,1]])
charge = np.ones(2)
print(ionsim.calculate_coulombpotential(r, charge))
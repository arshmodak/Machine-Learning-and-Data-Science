# -*- coding: utf-8 -*-
"""
Created on Sun Jun 20 13:48:49 2021

@author: Arsh Modak
"""

#%%

import numpy as np
from scipy import linalg

#%%

# M = np.array([[1, 2, 3],
#               [3, 4, 5],
#               [5, 4, 3],
#               [0, 2, 4],
#               [1, 3, 5]])

#%%

# USING NUMPY 

M = np.array([[3, 2, 2],
              [2, 3, -2]])

#%%


Mt = M.T

MTM = Mt@M

MMT = M@Mt

#%%

eigenvals, eigenvecs = np.linalg.eig(MTM)
print(eigenvals)
print(eigenvecs)

# print(np.linalg.norm(eigenvecs[:, 2]))

#%%

eigenvals, eigenvecs = np.linalg.eig(MMT)
print(eigenvals)
print(eigenvecs)

#%%

U, s, Vt = np.linalg.svd(M)
print(U)
print(s)
print(Vt)

#%%

print(eigenvecs/np.linalg.norm(eigenvecs))

#%%

eigenvals, eigenvecs = np.linalg.eig(MTM)
print(eigenvals)
print(eigenvecs)

eigenvals, eigenvecs = scipy.linalg.eig(MTM)
print(eigenvals)
print(eigenvecs)


#%%

# USING SCIPY
import scipy
np.linalg.svd(M)
#%%
from functools import partial
from math import pi
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)
import matplotlib.pyplot as plt

from qiskit.quantum_info import SparsePauliOp


def sparse_pauli_hamiltonian(nsites, edges, hfield=0):
    J = np.ones(len(edges)) 
    String = "I" * nsites
    H = SparsePauliOp("I" * nsites, 0)
    for j, c in zip(J, edges):
        c = np.sort(c)
        XX = String[: c[0]] + "X" + String[c[0] + 1 : c[1]] + "X" + String[c[1] + 1 :]
        H += SparsePauliOp(XX, j)
        YY = String[: c[0]] + "Y" + String[c[0] + 1 : c[1]] + "Y" + String[c[1] + 1 :]
        H += SparsePauliOp(YY, j)
        ZZ = String[: c[0]] + "Z" + String[c[0] + 1 : c[1]] + "Z" + String[c[1] + 1 :]
        H += SparsePauliOp(ZZ, j)
    
    # external field
    for i in range(nsites):
        H += SparsePauliOp("I"*i + "Z" + "I"*(nsites-i-1), -hfield)
    
    return H.chop()


def number_operator(nsites):
    N = SparsePauliOp("I"*nsites, nsites/2)
    for i in range(nsites):
        N += SparsePauliOp("I"*i + "Z" + "I"*(nsites-i-1), 0.5)
    return N


def n_of_h(h, nsites, edges):
    H = sparse_pauli_hamiltonian(nsites, edges, h)
    N = number_operator(nsites)

    Hmat = H.to_matrix()
    Nmat = N.to_matrix()
    eigvals, eigvecs = np.linalg.eigh(Hmat)

    igs = eigvals.argmin()
    gsval = eigvals[igs]
    gsvec = eigvecs[:, igs]
    spin = (gsvec.T@Nmat@gsvec).real 

    return gsval, spin



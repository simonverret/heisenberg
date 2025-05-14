from functools import partial
from math import pi
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=np.inf)
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.synthesis import LieTrotter, SuzukiTrotter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp


def heisenberg_hamiltonian(nsites, edges, hfield=0):
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
    H = heisenberg_hamiltonian(nsites, edges, h)
    N = number_operator(nsites)

    Hmat = H.to_matrix()
    Nmat = N.to_matrix()
    eigvals, eigvecs = np.linalg.eigh(Hmat)

    igs = eigvals.argmin()
    gsval = eigvals[igs]
    gsvec = eigvecs[:, igs]
    spin = (gsvec.T@Nmat@gsvec).real 

    return gsval, spin

fig, ax = plt.subplots(2, 1, figsize=(4,5), sharex=True)
fig.suptitle("$J\\sum_{\\langle ij \\rangle} S_i \\cdot S_j - h \\sum_{i} Z_i$")

for n, e in [
    (4, [(0,1), (2,3), (1,2), (3,0)]), 
    (6, [(0,1), (1,2), (2,3), (3,4), (4,5), (5,0)]), 
    (8, [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,0)]),
    # (10, [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9), (9,0)]),  # too long
]:
    print(n)

    n_of_h_4 = partial(n_of_h, nsites=n, edges=e)
    h_vec = np.linspace(-6,6, 200)
    e_vec = np.array([n_of_h_4(h)[0] for h in h_vec])
    s_vec = np.array([n_of_h_4(h)[1] for h in h_vec])

    ax[0].plot(h_vec, e_vec, label=f"{n} qbits")
    ax[1].plot(h_vec, s_vec, label=f"{n} qbits")

ax[0].set_ylabel("$E_0$")
ax[1].set_ylabel("$\\langle \\hat N \\rangle _{\\rm GS}$")
ax[1].set_xlabel("$h$")
ax[1].legend()
plt.tight_layout()
plt.show()


#%%
H = heisenberg_hamiltonian(4, [(0,1), (2,3), (1,2), (3,0)])
qc = QuantumCircuit(4)
qc.append(PauliEvolutionGate(H, pi/40, synthesis=LieTrotter(reps = 1)), range(n))
qc.decompose().draw('mpl', fold=-1)    
plt.show()

qc = QuantumCircuit(4)
qc.append(PauliEvolutionGate(H, pi/40, synthesis=SuzukiTrotter(reps = 1)), range(n))
qc.decompose().draw('mpl', fold=-1)    
plt.show()


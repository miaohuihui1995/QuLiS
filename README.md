# QuLiS: Quantum Lindblad Solver for Bosons

**0.Introduction**

QuLiS is a software package for solving the **Lindblad equation for bosonic systems**. It is built upon a reduced density matrix construction method, a ''point-determinant'' operation for the non-unitary term of the Lindblad equation, and a block-diagonalization method for the Hamiltonian and density matrix under the Markov approximation. It is suitable for solving the Lindblad equation whose **jump operators are bosonic annihilation operators**.

**1.Overview**

QuLiS targets the numerical simulation of dissipative dynamics of bosonic open quantum systems. Its core object is the Lindblad master equation of the form $\dot{\hat{\rho}}=-\frac{i}{\hbar}[\hat{H},\hat{\rho}]+\sum_k\left(\hat{L}_k\hat{\rho}\hat{L}_k^{\dagger}-\frac{1}{2}\lbrace\hat{L}_k^{\dagger}\hat{L}_k,\hat{\rho}\rbrace\right)$, where the jump operators $\hat{L}_k$ are bosonic annihilation operators. The package achieves efficient solutions through the following three mathematical methods:
- a reduced density matrix construction method;
- a ''point-determinant'' operation for the non-unitary term of the Lindblad equation;
- a block-diagonalization method for the Hamiltonian and density matrix under the Markov approximation.

**2.Key Advantages**

1. **Significantly reduced memory cost**: it greatly saves the memory overhead required to construct the density matrix and the Hamiltonian.
2. **Substantially improved computational efficiency**: it greatly improves the efficiency of solving the Lindblad equation.
3. **Accurate results without approximations**: none of the three mathematical methods involves any approximation; they do not alter the accuracy of the results, nor do they compromise the precision or convergence of numerical experiments.

**3.Applicable Scenarios**

- Dynamics of bosonic open quantum systems;
- Lindblad equations whose jump operators are bosonic annihilation operators;
- Dissipative evolution under the Markov approximation;
- Numerical experiments requiring efficient handling of large-scale density matrices and Hamiltonians.

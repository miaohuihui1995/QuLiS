# QuLiS: Quantum Lindblad Solver for Bosons

\section{Introduction}

\QuLiS{} is a software package for solving the \textbf{Lindblad equation for bosonic systems}.
It is built upon a reduced density matrix construction method, a ``point-determinant'' operation
for the non-unitary term of the Lindblad equation, and a block-diagonalization method for the
Hamiltonian and density matrix under the Markov approximation. It is suitable for solving the
Lindblad equation whose \textbf{jump operators are bosonic annihilation operators}.

\section{Overview}

\QuLiS{} targets the numerical simulation of dissipative dynamics of bosonic open quantum systems.
Its core object is the Lindblad master equation of the form
'''
  \dot{\rho}
  = -\frac{i}{\hbar}\,[H, \rho]
  + \sum_k \left(
      L_k \rho L_k^{\dagger}
      - \frac{1}{2}\{L_k^{\dagger} L_k, \rho\}
    \right),
'''
where the jump operators $L_k$ are bosonic annihilation operators. The package achieves
efficient solutions through the following three mathematical methods:
\begin{itemize}[leftmargin=1.5em]
  \item a reduced density matrix construction method;
  \item a ``point-determinant'' operation for the non-unitary term of the Lindblad equation;
  \item a block-diagonalization method for the Hamiltonian and density matrix under the
        Markov approximation.
\end{itemize}

\section{Key Advantages}

\begin{enumerate}[leftmargin=1.5em]
  \item \textbf{Significantly reduced memory cost}: it greatly saves the memory overhead
        required to construct the density matrix and the Hamiltonian.
  \item \textbf{Substantially improved computational efficiency}: it greatly improves the
        efficiency of solving the Lindblad equation.
  \item \textbf{Accurate results without approximations}: none of the three mathematical
        methods involves any approximation; they do not alter the accuracy of the results,
        nor do they compromise the precision or convergence of numerical experiments.
\end{enumerate}

\section{Applicable Scenarios}

\begin{itemize}[leftmargin=1.5em]
  \item Dynamics of bosonic open quantum systems;
  \item Lindblad equations whose jump operators are bosonic annihilation operators;
  \item Dissipative evolution under the Markov approximation;
  \item Numerical experiments requiring efficient handling of large-scale density matrices
        and Hamiltonians.
\end{itemize}

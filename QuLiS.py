# -*- coding:utf-8 -*-
import numpy as np
import time
import math
import copy


##################################################
##########      Matrix Exponential      ##########
##################################################


def PTSIM(H_m, NP, s):

    # Function: Compute the matrix exponential using the precise time step integration method
    # Input:  H_m - matrix
    #         NP  - typically set to 20
    #         s   - time step
    # Output: matrix exponential

    n = len(H_m)
    T_1 = H_m * s / (2 ** NP)
    T_2 = np.dot(T_1, T_1)
    T_3 = np.dot(T_2, T_1)
    T_4 = np.dot(T_3, T_1)
    T = T_1 + T_2 / 2. + T_3 / 6. + T_4 / 24.
    for i in range(NP):
        T = 2. * T + np.dot(T, T)

    return np.eye(n) + T


##################################################
######  BlockDiag of the Lindblad Equation  ######
##################################################


def GetOld2New(divisions, matrix_max = None):

    # Function: Construct the mapping from the global coordinates of quantum states on the density matrix
    #           to the local coordinates on the block-diagonal density matrix
    #           (including grouping + reordering)
    # Input:  divisions  - the label corresponding to each state (typically an energy label).
    #                     A 1D array: one-to-one correspondence with the label of each state.
    #         matrix_max - the maximum dimension of the matrix that each CPU can handle
    #                     (default: 4096).
    # Output: old2new    - the collection of mappings from global coordinates to local coordinates.
    #                     A 2D array: the index along the first dimension corresponds to different
    #                     block-diagonal density matrices; the data stored along the second dimension
    #                     are the global coordinates of the states within that block, while the index
    #                     along the second dimension is exactly the local coordinate of the state.
    #         ratio      - the memory optimization factor.
    #         areas      - the dimension of each block-diagonal density matrix.
    #                     A 1D array: one-to-one correspondence with the number of quantum states
    #                     contained in each block-diagonal density matrix.

    arr_unique = sorted(set(divisions), reverse = True) # Deduplicate + sort in descending order
    len_unique = len(arr_unique) # Number of block-diagonal density matrices
    old2new = []
    area_new = 0
    areas = []
    
    # Map the position of a quantum state on the diagonal of the density matrix (global index)
    # to its position within the block density matrix (block index + local index)
    for i in range(len_unique):
        index = []
        for j in range(len(divisions)):
            if(divisions[j] == arr_unique[i]):
                index.append(j)
        old2new.append(index)
        area_new += len(index) ** 2
        areas.append(len(index))
    ratio = len(divisions) ** 2 / area_new

    # Comparison between the total memory occupied by the block-diagonal density matrices
    # and the maximum CPU memory
    if matrix_max is None:
        matrix_max = 4096
    if(area_new / (matrix_max ** 2) > 1):
        print("Distributed computing is needed (%d CPUs)! The largest matrix dimension is %d * %d" % (math.ceil(area_new / (4096 ** 2)), max(areas), max(areas)))

    return old2new, ratio, areas


def BlockDiag(old2new, initial_state, s, energies, interactions, dissipations, dissi_coherence = None):

    # Function: Perform block diagonalization on the Lindblad master equation
    #           (unitary term + non-unitary term)
    # Input:  old2new      - see above
    #         energies     - the set of energies of the quantum states (when block diagonalization
    #                        is performed using energy labels, energies = divisions), used to fill
    #                        the diagonal elements of the block-diagonal Hamiltonian.
    #                        A 1D array: one-to-one correspondence with the energy of each state.
    #         interactions - the set of coupling terms (off-diagonal elements) of the block-diagonal
    #                        Hamiltonian. A 2D array: the index along the first dimension corresponds
    #                        to different coupling terms; the second dimension stores the global
    #                        x-coordinate, the global y-coordinate, and the value of the coupling term.
    #                        Note: this array does not store both a coupling term and its Hermitian
    #                        conjugate simultaneously; the Hermitian conjugate is obtained directly
    #                        later via the conjugate transpose.
    #         dissipations - the set of dissipation channels, used for the non-unitary term of the
    #                        Lindblad equation. A 2D array: the index along the first dimension
    #                        corresponds to different dissipation channels; the second dimension stores
    #                        the global coordinate of the starting quantum state of the dissipation
    #                        channel, the global coordinate of the ending quantum state, the
    #                        dissipation strength, and the inflow strength.
    #         initial_state - used to construct the initial block-diagonal density matrix (if the
    #                        initial state is a superposition state, all states participating in the
    #                        superposition must lie within the same block-diagonal density matrix, so
    #                        as to avoid inter-block coherence terms at the initial time).
    #                        A 1D array: the vector form of the quantum state.
    #         s = (t / iteration) / hbar, where t is the total duration, (t / iteration) is the time
    #                        step, and hbar is the reduced Planck constant, typically hbar = 1.
    # Output: exp_H_p, exp_H_m - the block-diagonalized unitary term. A 2D array: the index along the
    #                        first dimension corresponds to different blocks; the second dimension
    #                        stores exp_H_p and exp_H_m of different blocks.
    #         rho_0        - the block-diagonalized initial density matrix. A 2D array: the index along
    #                        the first dimension corresponds to different blocks; the second dimension
    #                        stores the initial density matrix of different blocks.
    #         L_parameters - the block-diagonalized set of dissipation channels (non-unitary term).
    #                        A 2D array: the index along the first dimension corresponds to different
    #                        dissipation channels; the second dimension stores the block coordinate of
    #                        the starting point of the dissipation channel, the local coordinate of the
    #                        starting point, the block coordinate of the ending point, the local
    #                        coordinate of the ending point, the dissipation strength, and the inflow
    #                        strength.

    start = time.time()

    # Generate all parameters for constructing the initial density matrix from initial_state
    rho_0_indexes = []
    for i in range(len(initial_state)):
        for j in range(len(initial_state)):
            if(initial_state[i] != 0 and initial_state[j] != 0):
                rho_0_indexes.append([i, j, initial_state[i] * initial_state[j]])

    # Perform block diagonalization on the unitary term and the initial density matrix
    exp_H_m = []
    exp_H_p = []
    rho_0 = []
    for i in range(len(old2new)):

        # Perform block diagonalization on the unitary term
        H_m_tmp = np.zeros((len(old2new[i]), len(old2new[i])))
        # Fill in the coupling terms of each block-diagonal Hamiltonian
        for j in range(len(interactions)):
            for k1 in range(len(old2new[i])):
                for k2 in range(len(old2new[i])):
                    if(interactions[j][0] == old2new[i][k1] and interactions[j][1] == old2new[i][k2]):
                        H_m_tmp[k1][k2] = interactions[j][2]
        H_m_tmp += H_m_tmp.T.conjugate() # conjugate transpose
        # Fill in the energy terms of each block-diagonal Hamiltonian
        for j in range(len(old2new[i])):
            H_m_tmp[j][j] = energies[old2new[i][j]]
        # Compute the matrix exponential
        exp_H_m_tmp = PTSIM(-1j * H_m_tmp, 20, s)
        exp_H_p_tmp = PTSIM(1j * H_m_tmp, 20, s)
        # Save
        exp_H_m.append(exp_H_m_tmp)
        exp_H_p.append(exp_H_p_tmp)

        # Perform block diagonalization on the initial density matrix
        rho_0_tmp = np.zeros((len(old2new[i]), len(old2new[i])))
        for j in range(len(rho_0_indexes)):
            for k1 in range(len(old2new[i])):
                for k2 in range(len(old2new[i])):
                    if(rho_0_indexes[j][0] == old2new[i][k1] and rho_0_indexes[j][1] == old2new[i][k2]):
                        rho_0_tmp[k1][k2] = rho_0_indexes[j][2]
        rho_0.append(rho_0_tmp)

    # Perform block diagonalization on the dissipation channels
    L_parameters = []
    for i in range(len(dissipations)):
        parameters = np.array([-1, -1, -1, -1, -1, -1])
        parameters[4] = dissipations[i][2]
        parameters[5] = dissipations[i][3]
        # Obtain the block coordinates and the local coordinates within the block
        # for the starting point and the ending point of each dissipation channel
        for j in range(len(old2new)):
            for k in range(len(old2new[j])):
                if(dissipations[i][0] == old2new[j][k]):
                    parameters[0] = j
                    parameters[1] = k
                if(dissipations[i][1] == old2new[j][k]):
                    parameters[2] = j
                    parameters[3] = k
        L_parameters.append(parameters)

    if dissi_coherence is None:

        end = time.time()
        # Measure the time taken for block diagonalization
        print("Block Diagonalization --- time: %d h %d m %d s" % (((end - start) // 3600), (((end - start) % 3600) // 60), (((end - start) % 3600) % 60)))

        return exp_H_m, exp_H_p, rho_0, L_parameters

    else:

        # Perform block diagonalization on the coherence terms between dissipation channels
        L_coherence = []
        for i in range(len(dissi_coherence)):
            parameter_coherence = np.array([-1., -1., -1., -1., -1., -1., -1., -1.])
            for j in range(len(old2new)):
                # Obtain the block coordinates and the local coordinates within the block
                # for the coherence terms between the starting points (ending points) of every two dissipation channels
                for k in range(len(old2new[j])):
                    if(dissi_coherence[i][0] == old2new[j][k]):
                        parameter_coherence[0] = j
                        parameter_coherence[1] = k
                    if(dissi_coherence[i][1] == old2new[j][k]):
                        parameter_coherence[2] = k
                    if(dissi_coherence[i][2] == old2new[j][k]):
                        parameter_coherence[3] = j
                        parameter_coherence[4] = k
                    if(dissi_coherence[i][3] == old2new[j][k]):
                        parameter_coherence[5] = k
                parameter_coherence[6] = dissi_coherence[i][4]
                parameter_coherence[7] = dissi_coherence[i][5]
            L_coherence.append(parameter_coherence)

        end = time.time()
        # Measure the time taken for block diagonalization
        print("Block Diagonalization --- time: %d h %d m %d s" % (((end - start) // 3600), (((end - start) % 3600) // 60), (((end - start) % 3600) % 60)))

        return exp_H_m, exp_H_p, rho_0, L_parameters, L_coherence


##################################################
### Block-Diagonalization-Based Time Evolution ###
##################################################


def Unitary(A, B, C):

    # Function: Solve the unitary term of the Lindblad master equation
    # Input:  A = exp_H_m, see above
    #         B = rho, the collection of block-diagonal density matrices
    #         C = exp_H_p, see above
    # Output: result - the collection of block-diagonal density matrices
    #                  after processing by the unitary term

    result = [] 
    for i in range(len(A)):
        result.append(np.dot(A[i], np.dot(B[i], C[i]))) # Unitary evolution
    return result


def NonUnitary(rho, s, L_parameters, L_coherence = None):

    # Function: Solve the non-unitary term of the Lindblad master equation
    # Input:  rho          - the collection of old block-diagonal density matrices
    #         s            - see above
    #         L_parameters - see above
    #         L_coherence  - used to compensate for the fact that L_parameters ignores the
    #                        coherence terms between dissipation channels when the number of
    #                        identical photons is greater than 1.
    #                        A 2D array: the index along the first dimension corresponds to different
    #                        pairs of dissipation channels; the second dimension stores the block
    #                        coordinates of the two starting points of the dissipation channel pair,
    #                        the local coordinate of starting point 1, the local coordinate of
    #                        starting point 2, the block coordinates of the two ending points,
    #                        the local coordinate of ending point 1, the local coordinate of
    #                        ending point 2, the dissipation strength, and the inflow strength.
    # Output: rho_new      - the collection of block-diagonal density matrices
    #                        after processing by the non-unitary term

    rho_new = copy.deepcopy(rho) # A copy of rho; all changes to the block-diagonal density matrices
                                 # caused by the non-unitary term are applied to rho_new

    # Handle the effect of dissipation channels on the block-diagonal density matrices
    for i in range(len(L_parameters)):
        if(L_parameters[i][4] == 0): # No dissipation, no inflow
            continue
        else:
            if(L_parameters[i][5] == 0): # Dissipation present, no inflow
                rho_new[L_parameters[i][2]][L_parameters[i][3]][L_parameters[i][3]] += L_parameters[i][4] *  rho[L_parameters[i][0]][L_parameters[i][1]][L_parameters[i][1]] * s
                for j in range(len(rho[L_parameters[i][0]])):
                    rho_new[L_parameters[i][0]][L_parameters[i][1]][j] -= 0.5 * L_parameters[i][4] * rho[L_parameters[i][0]][L_parameters[i][1]][j] * s 
                    rho_new[L_parameters[i][0]][j][L_parameters[i][1]] -= 0.5 * L_parameters[i][4] * rho[L_parameters[i][0]][j][L_parameters[i][1]] * s 
            else: # With dissipation, with inflow
                rho_new[L_parameters[i][2]][L_parameters[i][3]][L_parameters[i][3]] += L_parameters[i][4] *  rho[L_parameters[i][0]][L_parameters[i][1]][L_parameters[i][1]] * s
                rho_new[L_parameters[i][0]][L_parameters[i][1]][L_parameters[i][1]] += L_parameters[i][5] *  rho[L_parameters[i][2]][L_parameters[i][3]][L_parameters[i][3]] * s
                for j in range(len(rho[L_parameters[i][0]])):
                    rho_new[L_parameters[i][0]][L_parameters[i][1]][j] -= 0.5 * L_parameters[i][4] * rho[L_parameters[i][0]][L_parameters[i][1]][j] * s 
                    rho_new[L_parameters[i][0]][j][L_parameters[i][1]] -= 0.5 * L_parameters[i][4] * rho[L_parameters[i][0]][j][L_parameters[i][1]] * s 
                for j in range(len(rho[L_parameters[i][2]])):
                    rho_new[L_parameters[i][2]][L_parameters[i][3]][j] -= 0.5 * L_parameters[i][5] * rho[L_parameters[i][2]][L_parameters[i][3]][j] * s 
                    rho_new[L_parameters[i][2]][j][L_parameters[i][3]] -= 0.5 * L_parameters[i][5] * rho[L_parameters[i][2]][j][L_parameters[i][3]] * s 

    if L_coherence is not None:
        for i in range(len(L_coherence)):
            # Handle the effect of coherence terms between dissipation channels
            # on the block-diagonal density matrices
            if(L_coherence[i][6] == 0): # No dissipation, no inflow
                continue
            else:
                if(L_coherence[i][7] == 0): # Dissipation present, no inflow
                    rho_new[int(L_coherence[i][3])][int(L_coherence[i][4])][int(L_coherence[i][5])] += L_coherence[i][6] * rho[int(L_coherence[i][0])][int(L_coherence[i][1])][int(L_coherence[i][2])] * s
                    rho_new[int(L_coherence[i][3])][int(L_coherence[i][5])][int(L_coherence[i][4])] += L_coherence[i][6] * rho[int(L_coherence[i][0])][int(L_coherence[i][2])][int(L_coherence[i][1])] * s
                else: # With dissipation, with inflow
                    rho_new[int(L_coherence[i][3])][int(L_coherence[i][4])][int(L_coherence[i][5])] += L_coherence[i][6] * rho[int(L_coherence[i][0])][int(L_coherence[i][1])][int(L_coherence[i][2])] * s
                    rho_new[int(L_coherence[i][3])][int(L_coherence[i][5])][int(L_coherence[i][4])] += L_coherence[i][6] * rho[int(L_coherence[i][0])][int(L_coherence[i][2])][int(L_coherence[i][1])] * s
                    rho_new[int(L_coherence[i][0])][int(L_coherence[i][1])][int(L_coherence[i][2])] += L_coherence[i][7] * rho[int(L_coherence[i][3])][int(L_coherence[i][4])][int(L_coherence[i][5])] * s
                    rho_new[int(L_coherence[i][0])][int(L_coherence[i][2])][int(L_coherence[i][1])] += L_coherence[i][7] * rho[int(L_coherence[i][3])][int(L_coherence[i][5])][int(L_coherence[i][4])] * s

    return rho_new


##################################################
##########       Obtain the Result      ##########
##################################################


def GetRhoDiag(rho_diag, rho, old2new, n):

    # Function: Obtain the populations on the diagonal of the density matrix
    # Input:  rho_diag - used to store the populations on the diagonal.
    #                     A 2D array: the index along the first dimension corresponds to
    #                     each iteration in order; the second dimension stores the populations
    #                     on the diagonal (in global-coordinate order).
    #         rho      - the collection of the latest block-diagonal density matrices
    #                     obtained at each iteration
    #         old2new  - see above
    #         n        - the total number of quantum states
    # Output: rho_diag
    
    diagonal = np.zeros(n)
    for i in range(len(rho)):
        for j in range(len(rho[i])):
            diagonal[old2new[i][j]] = np.maximum(np.real(rho[i][j][j]), 0) # The reason for taking the real part and discarding negative real parts here is that,
                                                                           # in numerical computation, extremely small imaginary parts and negative real parts
                                                                           # can arise due to floating-point errors.
    #        diagonal[old2new[i][j]] = np.abs(rho[i][j][j]) # Here we take the absolute value. However, since the imaginary part is included,
                                                            # the populations will be slightly higher than if we simply took the real part.
    rho_diag.append(diagonal)
    return rho_diag


def SparseVecMatVec(rho_i, state_i):

    # Function: Efficiently compute <state_i|rho_i|state_i> by exploiting sparsity
    # Input:  rho_i   - a single block-diagonal density matrix
    #         state_i - the slice of the target state within the corresponding block
    # Output: expectation.real.item() - the expectation value

    # Find the nonzero indices
    idx = np.nonzero(state_i)[0]
    
    if len(idx) == 0:
        return 0.

    else:
    
        # Take only the nonzero part
        state_i_sparse = state_i[idx]
    
        # Compute rho_i_sparse = rho_i[idx][:, idx] (take only the nonzero rows and columns)
        rho_i_sparse = rho_i[np.ix_(idx, idx)]
    
        # Compute <state_i_sparse|rho_i_sparse|state_i_sparse>
        expectation = np.dot(state_i_sparse.conj(), np.dot(rho_i_sparse, state_i_sparse))
    
        return expectation.real.item() # Convert to a scalar and discard the imaginary part produced by floating-point errors


def GetRhoState(rho_state, rho, old2new, state):

    # Function: Obtain the population of a certain state (possibly a superposition state)
    #           on the density matrix
    # Input:  rho_state - used to store the population of a certain state.
    #                     A 2D array: the index along the first dimension corresponds to
    #                     each iteration in order; the second dimension stores the population
    #                     of that state.
    #         rho       - the collection of the latest block-diagonal density matrices
    #                     obtained at each iteration
    #         old2new   - see above
    #         state     - the quantum state. A 1D array: one-to-one correspondence with the
    #                     value on each basis vector (in global-coordinate order)
    # Output: rho_state

    # Block-diagonalize the quantum state
    state_blocks = []
    for i in range(len(old2new)):
        block_tmp = []
        for j in range(len(old2new[i])):
            block_tmp.append(state[old2new[i][j]])
        state_blocks.append(np.array(block_tmp))

    # <state|rho|state>
    expectation = 0.
    for i in range(len(rho)):
        expectation += SparseVecMatVec(rho[i], state_blocks[i])
    
    # Squared modulus
    norm_sq = np.dot(state.reshape(-1, 1).conj().T, state.reshape(-1, 1)).item() 

    rho_state.append(expectation / norm_sq if norm_sq > 1e-12 else 0.0)

    return rho_state

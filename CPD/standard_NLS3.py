import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

def jacobian(A):
    assert(len(A)==3,"This function only works for 3D tensors.")
    J = np.zeros((Q,order*s*R))
    for i in range(order):
        offset1 = i*s*R
        for j in range(R):
            offset2 = j*s
            start = offset1+offset2
            end = start + s
            if i==0:
                J[:,start:end] = np.kron(np.identity(s),np.kron(A[1][:,j],A[2][:,j])).T
            elif i==1:
                J[:,start:end] = np.kron(A[0][:,j],np.kron(np.identity(s),A[2][:,j])).T
            elif i==2:
                J[:,start:end] = np.kron(A[0][:,j],np.kron(A[1][:,j],np.identity(s))).T
    return J

def gradient(T,A):
    assert(len(A)==3,"This function only works for 3D tensors.")
    g = np.zeros(order*s*R)
    G = []

    TC = tenpy.einsum("ijk,ka->ija",T,A[2])
    M1 = tenpy.einsum("ija,ja->ia",TC,A[1])
    G1 = -1*M1 + np.dot(A[0],compute_lin_sys(tenpy,A[1],A[2],0))
    G.append(G1)

    M2 = tenpy.einsum("ija,ia->ja",TC,A[0])
    G2 = -1*M2 + np.dot(A[1],compute_lin_sys(tenpy,A[0],A[2],0))
    G.append(G2)

    M3 = tenpy.einsum("ijk,ia,ja->ka",T,A[0],A[1])
    G3 = -1*M3 + np.dot(A[2],compute_lin_sys(tenpy,A[0],A[1],0))
    G.append(G3)
    
    return G

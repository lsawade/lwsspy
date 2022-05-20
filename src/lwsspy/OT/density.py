from numba import njit, prange
from numba import jit

@jit(nopython=True, parallel=True)
def dpnqr_duk(pij, dpij_duk):
    
    # Getting the arrays
    dpnqr_duk_arr = np.zeros_like(dpij_duk)
    
    # Normalizing factors
    Np = np.sum(pdij)
    
    # Normalize
    pij_norm = pij/Np
    
    # Get number of rows and colums
    M, N = pij.shape
    
    for q in prange(M):
        for r in range(N):
            tmp = 0
            
            for i in range(M):
                for j in range(N):
                    if i==q and j == r:
                        tmp += (1 - pij_norm[q,r])/Np * dpij_duk[i,j]
                    else:
                        tmp += (-pij_norm[q,r])/Np * dpij_duk[i,j]
                        
            dpnqr_duk_arr[q,r] = tmp
            
    return dpnqr_duk_arr
                    
    
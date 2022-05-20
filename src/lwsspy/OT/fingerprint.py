import numpy as np
from numba import njit, prange
from numba import jit

@jit(nopython=True, parallel=True)
def fingerprint(u, t):
    
    # Define p grid
    pu = np.linspace(0, 1, 150)
    pt = np.linspace(np.min(t), np.max(t), 100)
    
    # Get length:
    nu = len(pu)
    nt = len(pt)
    nk = len(u)-1
    
    # Initialize d
    d = np.zeros((nt, nu), dtype=np.float64)
        
    # Initialize arrays 
    l = np.zeros(nk, dtype=np.float64)
    x0 = np.zeros(2, dtype=np.float64)
    x1 = np.zeros(2, dtype=np.float64)
    p = np.zeros(2, dtype=np.float64)
    
    # Loop over grid points and segments
    for i in prange(nt):
        for j in range(nu):
            
            # Reassign p
            p[0] = pt[i]
            p[1] = pu[j]
            
            # Zero out l
            l = np.zeros(nk, dtype=np.float64)
            
            for k in range(nk):
                # Get segment coords
                x0[0] = t[k]
                x0[1] = u[k]
                x1[0] = t[k+1]
                x1[1] = u[k+1]
                
                # Get lambda
                dx = x1 - x0
                dx2 = dx[0]**2 + dx[1]**2
                lmd = ( (p[0] - x0[0])*dx[0] + (p[1] - x0[1])*dx[1] )/ dx2
                
                if lmd>1:
                    lmd = 1
                elif lmd<0:
                    lmd=0
                    
                # Get local dist
                dpx = p - ((1-lmd)*x0 + lmd*x1)
                dpx2 = dpx[0]**2 + dpx[1]**2
                l[k] = np.sqrt(dpx2)
                
            d[i, j] = np.min(l)

            
    return pt, pu, d


@jit(nopython=True, parallel=True)
def fingerprint_and_diff(u, t):
    
    # Define p grid
    pu = np.linspace(0, 1, 150)
    pt = np.linspace(np.min(t), np.max(t), 100)
    
    # Get length:
    nu = pu.size
    nt = pt.size
    nk = u.size-1
    
    # Initialize d
    d = np.zeros((nt, nu), dtype=np.float64)
    dij_k = np.zeros((nt, nu), dtype=np.float64)
    dij_kp1 = np.zeros((nt, nu), dtype=np.float64)
    lmbdas = np.zeros((nt, nu), dtype=np.float64)
    idk = np.zeros((nt, nu), dtype=np.int64)
    
    # Initialize arrays 
    x0 = np.zeros(2, dtype=np.float64)
    x1 = np.zeros(2, dtype=np.float64)
    p = np.zeros(2, dtype=np.float64)
    
    # Loop over grid points and segments
    for i in prange(nt):
        for j in range(nu):
            
            # Reassign p
            p[0] = pt[i]
            p[1] = pu[j]
            
            # Zero out l, lmds
            l = np.zeros(nk, dtype=np.float64)
            lmds = np.zeros(nk, dtype=np.float64)
            
            for k in range(nk):
                # Get segment coords
                x0[0] = t[k]
                x0[1] = u[k]
                x1[0] = t[k+1]
                x1[1] = u[k+1]
                
                # Get lambda
                dx = x1 - x0
                dx2 = dx[0]**2 + dx[1]**2
                lmds[k] = ( (p[0] - x0[0])*dx[0] + (p[1] - x0[1])*dx[1] )/ dx2
                
                if lmds[k] > 1:
                    lmds[k] = 1
                elif lmds[k] < 0:
                    lmds[k] = 0

                # Get local dist
                dpx = p - ((1-lmds[k])*x0 + lmds[k]*x1)
                dpx2 = dpx[0]**2 + dpx[1]**2
                l[k] = np.sqrt(dpx2)

                
            # C.5
            kl = np.argmin(l)
            d[i, j] = l[kl]
            lmbdas[i,j] = lmds[kl]
            idk[i,j] = kl
            
            uclose = u[k] + dx[1] * l[kl]
            
            # C.9 Error in this equation it should be u(lambda)
            dij_k[i, j] = (1-lmds[kl])/d[i,j] * (uclose - pu[j])
            
            # C.10 Error in this equation it should be u(lambda)
            dij_kp1[i, j] = lmds[kl] * (uclose - pu[j])/d[i,j]
            
    return pt, pu, d, lmbdas, dij_k, dij_kp1, idk


def VMconst(np,npl,epsEtr,mCst):
    """
    von Mises linear elastic perfectly plastic constitutive model

    Description:
    -----------
    von Mises perfect plasticity constitutive model with an implicit backward \
Euler stress integration algorithm based on the following thesis:
    
    Coombs, W.M. (2011). Finite deformation of particulate geomaterials: \
frictional and anisotropic Critical State elasto-plasticity. School of \
Engineering and Computing Sciences. Durham University. PhD.
    
    Parameters
    ----------
    np     : imported numpy module
    npl    : imported numpy.linalg module
    epsEtr : trial elastic strain (6,1)
    mCst   : material constants 

    Returns
    -------
    sig    : Cauchy stress (6,1)
    epsE   : elastic strain (6,1)
    Dalg   : algorithmic consistent tangent (6,6)
    
    Calling function:
    ----------------
    Dalg,sigma,epsE = VMconst(np,npl,epsEtr,mCst)
    
    See also:
    ---------
    yieldFuncDerivatives : yield function 1st and 2nd derivatives
    """    

    E = mCst[0];  v = mCst[1]; rhoY = mCst[2]                                   # material constants
    tol = 1e-9;   maxit = 5                                                     # NR parameters
    bm1 = np.array([1, 1, 1, 0, 0, 0]).reshape(6,1)                             # vector form of an identity matrix
    
    # python not as good as MATLAB in forming matrices - syntax bulky
    Ce = np.zeros((6,6))                                                        # create empty array for elastic compliance matrix
    Ce[0:3,0:3] = -v + (1+v)*np.identity(3)                                     # fill upper left hand side
    Ce[3:6,3:6] = 2*(1+v)*np.identity(3)                                        # fill lower right hand side
    Ce = Ce/E                                                                   # elastic compliance matrix
    
    De_diag = np.diag([1,1,1,0.5,0.5,0.5])*(1-2*v)
    De = E/((1+v)*(1-2*v))*(bm1*bm1.T*v + De_diag)                              # elastic stiffness matrix
    
    sig = De @ epsEtr                                                           # elastic trial stress
    s   = sig - np.sum(sig[0:3])/3*bm1                                          # deviatoric stress 
    j2  = (s.T @ s + s[3:6].T @ s[3:6])/2                                       # second invariant of devatoric stress (J2)
    f   = np.sqrt(2*j2)/rhoY -1                                                 # yield function
    epsE = epsEtr.copy(); Dalg = De.copy()                                      # set the elastic case
    
    if f > tol:                                                                 # plasticity loop
        b = np.zeros((7,1)); b[6] = f; itnum = 0; dgam = 0;                     # initial conditions on the NR search
        df,ddf = yieldFuncDerivatives(np,sig,rhoY)                              # 1st and 2nd derivative of f wrt. stress
        while (itnum < maxit) and ((npl.norm(b[0:6]) > tol) or (abs(b[6]) > tol)):      # NR loop
            
            A = np.zeros((7,7))                                                 # create empty array to fill with derivative of the residuals wrt. unknowns (Hessian)
            A[0:6,0:6] = np.identity(6) + dgam * ddf @ De
            A[:-1,6] = df.flatten()
            A[6,:-1] = df.T @ De
            dx   = - npl.inv(A) @ b                                             # increment unknowns (dx, epsE, dgam)
            epsE = epsE + dx[0:6]                                               
            dgam = dgam + dx[6]
            sig  = De @ epsE                                                    # updated stress
            s    = sig - np.sum(sig[0:3])/3*bm1                                 # deviatoric stress 
            j2   = (s.T @ s + s[3:6].T @ s[3:6])/2                              # second invariant of devatoric stress (J2)
            df,ddf = yieldFuncDerivatives(np,sig,rhoY)                          # 1st and 2nd derivative of f wrt. stress
            b_1    = epsE - epsEtr + dgam * df                                  # first term in residuals
            b_2    = np.sqrt(2*j2)/rhoY -1                                      # second term in residuals
            b      = np.vstack((b_1,b_2))                                       # residuals
            itnum  = itnum + 1
                 
        B = np.zeros((7,7))
        B[0:6,0:6] = Ce + dgam * ddf
        B[:-1,6] = df.flatten()
        B[6,:-1] = df.T                       
        B = npl.inv(B)                                                          # Aalg from eqn (2.53) of Coombs(2011) thesis
        Dalg = B[0:6,0:6]                                                       # algorithmic consistent tangent
    
    return Dalg,sig,epsE

def yieldFuncDerivatives(np,sig,rhoY):
    """
    von Mises yield function derivatives

    Description:
    -----------
    First and second derivatives of the von Mises yield function with respect \
to stress.
    
    Parameters
    ----------
    np    : imported numpy module
    sigma : Cauchy stress (6,1)
    rhoY  : von Mises yield strength (1)

    Returns
    -------
    df    : derivative of the yield function wrt. sigma (6,1)
    ddf   : second derivative of the yield function wrt. sigma (6,6)
    
    Calling function:
    ----------------
    df,ddf = yieldFuncDerivatives(np,sig,rhoY)
    """    
    
    bm1 = np.array([1, 1, 1, 0, 0, 0]).reshape(6,1)                             # vector form of an identity matrix
    s   = sig - np.sum(sig[0:3])/3*bm1                                          # deviatoric stress 
    j2  = (s.T @ s + s[3:6].T @ s[3:6])/2                                       # second invariant of devatoric stress (J2)
    dj2 = s.copy()                                                              # derivative of J2 wrt. stress
    dj2[3:6] = 2*dj2[3:6]               
    ddj2 = np.zeros((6,6))                                                      # create empty array to fill with second derivative of J2 wrt. stress
    ddj2[0:3,0:3] = np.identity(3) - 1/3                                        # fill upper left of ddj2
    ddj2[3:6,3:6] = np.identity(3)*2                                            # fill lower right of ddj2
    df  = dj2 / (rhoY*np.sqrt(2*j2))                                            # derivative of f wrt. stress
    ddf = (1/rhoY) * (ddj2/np.sqrt(2*j2) - (dj2*dj2.T) / (2*j2)**(3/2))         # 2nd derivative of f wrt. stress
    
    return df,ddf
def Hooke3d(np,epsE,mCst):
    """
    Linear elastic constituitive model

    Parameters
    ----------
    np   : import numpy module
    epsE : elastic strain (6,1)
    mCst : material constants 

    Returns
    -------
    D    : elastic stiffness matrix (6,6)
    sig  : stress (6,1)
    epsE : elastic strain (6,1)
    
    Calling function:
    ----------------
    D,sig,epsE = Hooke3d(np,epsE,mCst)
    """
    
    E = mCst[0]                                                                 # Young's modulus
    v = mCst[1]                                                                 # Poisson's ratio
    bm11 = np.array([[1,1,1,0,0,0],
                     [1,1,1,0,0,0],
                     [1,1,1,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0],
                     [0,0,0,0,0,0]])
    
    
    D = (E/((1+v)*(1-2*v))) * (bm11*v+np.diag([1, 1, 1, 0.5, 0.5, 0.5])*(1-2*v)) # elastic stiffness
    sig = D @ epsE                                                               # stress 

    return D, sig, epsE
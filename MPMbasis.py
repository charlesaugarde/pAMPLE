def MPMbasis(np,mesh,mp_data,nD,node):
    """
    Basis functions for the material point method

    Description:
    ------------
    Function to determine the multi-dimensional MPM shape functions from the\
one dimensional MPM functions.  The function includes both the standard\
and generalised interpolation material point methods. 
    
    Parameters
    ----------
    np      : import numpy module as np
    mesh    : input given in call is meshData object. Function requires:
                > coord  : nodal coordinates  
                > h      : grid spacing
    nD      : number of dimensions
    mp_data : input given in call is mp_data[mp] for mp in loop.  Function requires:
                > mpC    : material point coordinates (of single point)
                > lp     : particle domain lengths
                > mpType : material point type (1 or 2)
    node    : background mesh node number

    Returns
    -------
    Svp  : particle characteristic function
    dSvp : gradient of the characterstic function 

    Calling function:
    ----------------
    Svp,dSvp = MPMbasis(np,mesh,mp_data,nD,node)    
    
    See also:
    ---------
    SvpMPM  : MPM basis functions in 1D (mpType = 1)
    SvpGIMP : GIMPM basis functions in 1D (mpType = 2)
    """
    from SvpMPM import SvpMPM
    from SvpGIMP import SvpGIMP
    
    coord  = mesh.coord[node-1,:]                                               # node coordinates (-1 because of zero-indexing)
    h      = mesh.h                                                             # grid spacing
    mpC    = mp_data.mpC                                                        # material point coordinates
    lp     = mp_data.lp                                                         # material point domain length
    mpType = mp_data.mpType                                                     # material point type (MPM or GIMPM)
    
    S = np.zeros([nD,1])                                                        # zero vectors used in calcs
    dS = np.zeros([nD,1]) 
    dSvp = np.zeros([nD,1])
    
    for i in range(0,nD):                                                       
        if mpType == 1:
            S[i], dS[i] = SvpMPM(mpC[i],coord[i],h[i])                          # 1D MPM functions                   
        elif mpType == 2:
            S[i], dS[i] = SvpGIMP(mpC[i],coord[i],h[i],lp[i])                   # 1D GIMPM functions
        # in each case, it uses 1 element of 2x1 array at a time as test (mentioned above)
            
    if nD == 1:                                                                 # index for basis derivatives (1D)
        indx = np.array([])
    elif nD == 2:                                                               # index for basis derivatives (2D)
        indx = np.array([1,0]).reshape(2,1) 
    elif nD == 3:                                                               # index for basis derivatives (3D)
        indx = np.array([1,2,0,2,0,1]).reshape(3,2)
    # need indx-1 (compared to MATLAB code) because of zero indexing
    
    Svp = np.prod(S)                                                            # basis function. prod = product of array elements
    
    for i in range(0,nD):
        dSvp[i] = dS[i]*np.prod(S[indx[i,:]])                                   # gradient of the basis function
    
    return Svp,dSvp
def shapefunc(np,nen,GpLoc,nD):
    """
    Finite element basis functions
    
    Description:
    ------------
    Function to provide finite element shape functions in 1D, 2D and 3D.  The\
function includes the following elements:
    > nen = 8, nD = 3 : tri-linear eight noded hexahedral
    > nen = 4, nD = 2 : bi-linear four noded quadrilateral
    > nen = 2, nD = 1 : linear two noded line

    N = shapefunc(np,nen,GpLoc,nD)

    Parameters
    ----------
    np    : imports numpy module (as np)
    nen   : number of nodes associated with the element
    GpLoc : local position within the finite element (n,nD)
    nD    : number of dimensions

    Returns
    -------
    N : shape function matrix (n,nen)

    Calling function:
    ----------------
    N = shapefunc(np,nen,GpLoc,nD)
    """
    
    n = GpLoc.shape[0]                                                          # number of points
    N = np.zeros([n,nen])                                                       # zero shape function matrix

    if nD == 3:                                                                 # 3D
        xsi = GpLoc[:,0]; eta = GpLoc[:,1]; zet = GpLoc[:,2]                    # local position
        if nen == 8:                                                            # 8-noded hexahedral
            N[:,0]=(1-xsi)*(1-eta)*(1-zet)/8
            N[:,1]=(1-xsi)*(1-eta)*(1+zet)/8
            N[:,2]=(1+xsi)*(1-eta)*(1+zet)/8
            N[:,3]=(1+xsi)*(1-eta)*(1-zet)/8
            N[:,4]=(1-xsi)*(1+eta)*(1-zet)/8
            N[:,5]=(1-xsi)*(1+eta)*(1+zet)/8
            N[:,6]=(1+xsi)*(1+eta)*(1+zet)/8
            N[:,7]=(1+xsi)*(1+eta)*(1-zet)/8
    
    elif nD == 2:                                                               # 2D
        xsi = GpLoc[:,0]; eta = GpLoc[:,1]                                      # local positions
        if nen == 4:                                                            # 4-noded quadrilateral
            N[:,0]=(1-xsi)*(1-eta)/4
            N[:,1]=(1-xsi)*(1+eta)/4
            N[:,2]=(1+xsi)*(1+eta)/4
            N[:,3]=(1+xsi)*(1-eta)/4

    else:                                                                       # 1D
        xsi = GpLoc                                                             # local positions
        if nen == 2:                                                            # 2-noded line
            N[:,0]=(1-xsi)/2
            N[:,1]=(1+xsi)/2     

    return N
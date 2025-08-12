def elemForMP(np,coord,etpl,mpC,lp):
    """
    Find elements associated with the material point

    Description:
    ------------
    Function to determine the elements that are associated with a material \
point assuming that the material point's domain is symmetric about the \
particle position.
    
    Parameters
    ----------
    np    - imports numpy module (as np)
    coord - element coordinates (nen,nD)
    etpl  - element topology (nels,nen)
    mpC   - material point coordinates (1,nD)
    lp    - domain half width

    Returns
    -------
    eIN   - vector containing the elements associated with the mp
    
    Calling function:
    ----------------
    eIN = elemForMP(np,coord,etpl,mpC,lp)    
    """
    
    nD   = coord.shape[1]                                                       # number of dimensions
    nels = etpl.shape[0]                                                        # number of elements
    Pmin = (mpC-lp).flatten()                                                   # particle domain lengths (lower)
    Pmax = (mpC+lp).flatten()                                                   # particle domain lengths (upper)
    a    = np.ones((nels, 1), dtype=bool)                                       # initialises boolean logical array of size nelsx1

    for i in range(0,nD):                                                       
        ci = coord[:,i]                                                         # nodal coordinates in current i direction
        c  = ci[etpl-1]                                                         # reshaped element coordinates in current i direction (# -1 from etpl to account for python zero indexing)
        Cmin = (c.min(axis=1)).reshape(nels,1)                                  # element lower coordinate limit (returned as array containing minimum value of each row)    
        Cmax = (c.max(axis=1)).reshape(nels,1)                                  # element upper coordinate limit
        # For a 2D array, numbers.min() finds the single minimum value in 
        # the array, numbers.min(axis=0) returns the minimum value for each 
        # column and numbers.min(axis=1) returns the minimum value for each row.
        a = a * ((Cmin<Pmax[i]) * (Cmax>Pmin[i]))                               # element overlap with mp domain
        
    eIN = np.array([range(0,nels)]).reshape(nels,1)                             # list of all elements
    eIN = eIN[a>0].astype(int)                                                  # remove those elements not in the domain
    
    return eIN
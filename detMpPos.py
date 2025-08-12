def detMpPos(np,mp,nD):
    """
    Material point local positions for point generation
    
    Description:
    ------------
    Function to return the local positions of the material points for initial\
 set up of a problem.  The material points are evenly spaced through the \
 elements that can be regular line, quadrilateral and hexahedral elements \
 in 1D, 2D and 3D. 
    
    Parameters
    ----------
    np : imports numpy module (as np)
    mp : number of material points in each direction
    nD : number of dimensions
    
    Returns
    -------
    mpPos : local material point positions (nmp,nD)
    
    Calling function
    ----------------
    lstps,g,mp_data,meshData = setupGrid_collapse(np)
    """        
    
    nmp   = mp**nD                                                              # number of material points per element (set by analyst in setupGrid)
    mpPos = np.zeros([nmp,nD])                                                  # zero material point positions
    a     = 2/mp                                                                # local length associated with the material point (eta goes from -1 -> 1)
    b     = np.arange(a/2,2,a)-1                                                # local positions of MP in 1D
    
    if nD == 1:                                                                 # 1D
        mpPos = np.atleast_2d(b).T
    
    elif nD == 2:                                                               # 2D
        for i in range(0,mp):
            for j in range(0,mp):
                mpPos[(i)*mp+j][0] = b[i] 
                mpPos[(i)*mp+j][1] = b[j] 
    else:                                                                       # 3D
        for i in range(0,mp): 
            for j in range(0,mp):
                for k in range(0,mp):              
                    mpPos[(i-1)*mp**2+(j-1)*mp+k][0] = b[i] 
                    mpPos[(i-1)*mp**2+(j-1)*mp+k][1] = b[j] 
                    mpPos[(i-1)*mp**2+(j-1)*mp+k][2] = b[k]
    return mpPos
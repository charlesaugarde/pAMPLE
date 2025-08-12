def detFDoFs(np,npm,mesh):
    """
    Determine the free degrees of freedom on the background mesh       

    Description:
    ------------
    Function to determine the free degrees of freedom of the background mesh \
based on the elements that contain material points and the displacement \
boundary conditions. 

    Parameters
    ----------
    np   : imports numpy module (as np)
    npm  : imports numpy.matlib module (as npm)
    mesh : object of mesh class. Function requires attributes: 
              > etpl : element topology (nels,nen) 
              > eInA : elements "active" in the analysis
              > bc   : boundary conditions (*,2)

    Returns
    -------
    fd : free degrees of freedom on the background mesh (*,1)
    """
    
    nodes,nD = mesh.coord.shape                                                 # no. nodes and dimensions
    nDoF   = nodes*nD;                                                          # no. degrees of freedom
    incN   = np.unique(mesh.etpl[mesh.eInA[:,0]>0,:])                           # unique active node list
    # need mesh.eInA[:,0]>0 rather than mesh.eInA>0
    iN      = incN.shape[0]                                                     # number of nodes in the list
    incDoF1 = npm.repmat(incN*nD,nD,1) 
    # create 'row vector' (incN) that is copied nD times
    incDoF2 = np.arange(nD-1,-1,-1).reshape(nD,1)
    incDoF2 = npm.repmat(incDoF2,1,iN)
    # creates nDx1 column vector that is turned into row vector (iN long)
    incDoF  = (incDoF1 - incDoF2).flatten(order = 'F')                          # active degrees of freedom
    fd      = np.arange(1,nDoF+1)  #need +1 because of zero indexing            # all degrees of freedom
    fd[mesh.bc[:,0]-1] = 0                                                      # zero fixed displacement BCs
    #set fd entries corresponding to bc indexes (-1) to zero 
    fd     = fd[incDoF-1]                                                       # only include active DoF 
    fd     = fd[fd>0]                                                           # remove fixed displacement BCs

    return fd
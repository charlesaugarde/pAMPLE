def elemMPinfo(np,mp_data,meshData):
    """
    Determine the basis functions for material points 
    
    Description:
    ------------
    Function to determine the basis functions and spatial derivatives of each \
material point.  The function works for regular background meshes with \
both the standard and generalised interpolation material point methods.

The function also determines, and stores, the elements associated with \
the material point and a unique list of nodes that the material point \
influences.  The number of stiffness matrix entries for each material \
point is determined and stored. 
       
    Parameters
    ----------
    np       : imports numpy module (as np)
    mp_data  : list of material point objects (with attributes)
    meshData : mesh class object. Function requires: 
                > coord - coordinates of the grid nodes (nodes,nD)
                > etpl  - element topology (nels,nen) 
                > h     - background mesh size (nD,1)    
    
    Returns
    -------
    meshData : mesh class object. Function modifies:
                > eInA  - elements in the analysis 
    mp_data  : list of objects storing material point objects. Function modifies:
                > nIN  - nodes linked to the material point
                > eIN  - element associated with the material point
                > Svp  - basis functions for the material point
                > dSvp - basis function derivatives (at start of lstp)
                > nSMe - number stiffness matrix entries for the MP
    
    Calling function:
    ----------------
    meshData,mp_data = elemMPinfo(np,mp_data,meshData)
    
    See also:
    ---------
    elemForMP  : find elements for material point
    nodesForMP : nodes associated with a material point 
    MPMbasis   : MPM basis functions
    """
    
    from elemForMP import elemForMP
    from nodesForMP import nodesForMP
    from MPMbasis import MPMbasis

    nmp  = len(mp_data)                                                         # number of material points
    nD   = meshData.coord.shape[1]                                              # number of nodes (total in mesh)  
    nels = meshData.etpl.shape[0]                                               # number of elements in mesh
    eInA = np.zeros([nels,1])                                                   # zero elements taking part in the analysis
            
    for mp in range(0,nmp):
        eIN  = elemForMP(np,meshData.coord,meshData.etpl,\
                         mp_data[mp].mpC,mp_data[mp].lp)                        # attributes connected to the specified material point in loop
        nIN  = nodesForMP(np,meshData.etpl,eIN)                                 # unique list of nodes associated with elements
        nn   = len(nIN)                                                         # number of nodes influencing the MP
        Svp  = np.zeros([1,nn]).flatten('F')                                    # zero basis functions
        dSvp = np.zeros([nD,nn])                                                # zero basis function derivatives
            
        for i in range(0,nn): 
            node = nIN[i]                                                       # current node
            S,dS = MPMbasis(np,meshData,mp_data[mp],nD,node)                    # basis function and spatial derivatives
            Svp[i]    = Svp[i].copy() + S                                       # basis functions for all nodes
            dSvp[:,i] = dSvp[:,i].copy() + dS.flatten('F')                      # basis function derivatives
                
        mp_data[mp].nIN  = nIN                                                  # nodes associated with material point
        mp_data[mp].eIN  = eIN + 1                                              # elements associated with material point (need +1 because of zero indexing)
        mp_data[mp].Svp  = Svp                                                  # basis functions
        mp_data[mp].dSvp = dSvp                                                 # basis function derivatives
        mp_data[mp].nSMe = (nn*nD)**2                                           # number stiffness matrix components
        eInA[eIN] = 1                                                           # identify elements in the analysis

    meshData.eInA = eInA.astype(int)                                            # store eInA to meshData object
    
    return meshData,mp_data
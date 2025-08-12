def detExtForce(np,npm,nodes,nD,g,mp_data):
    """
    Global external force determination 
    
    Description:
    ------------
    Function to determine the external forces at nodes based on body forces \
and point forces at material points.
    
    Parameters
    ----------
    np      : import numpy module as np
    nodes   : number of nodes (total in mesh)
    nD      : number of dimensions
    g       : gravity
    mp_data : list of mpDataPoint objects. Function requires:
                > mpM   : material point mass
                > nIN   : nodes linked to the material point
                > Svp   : basis functions for the material point
                > fp    : point forces at material points

    Returns
    -------
    fext   - external force vector (nodes*nD,1)
    
    Calling function
    ----------------
    fbdy,mpData = detExtForce(np,npm,nodes,nD,g,mp_data)
    """
    
    nmp  = len(mp_data)                                                         # number of material points & dimensions 
    fext = np.zeros([nodes*nD,1])                                               # zero the external force vector
    grav = np.zeros([nD,1])                                                     # initialise gravity vector
    grav[nD-1] = -g                                                             # gravity vector
    
    for mp in range(0,nmp):
        nIN = mp_data[mp].nIN                                                   # nodes associated with MP
        nn  = len(nIN)                                                          # number of nodes influencing the MP
        Svp = mp_data[mp].Svp                                                   # basis functions
        fp  = (mp_data[mp].mpM*grav + mp_data[mp].fp)*Svp                       # material point body & point nodal forces
        fp  = fp.flatten('F').reshape(nn*nD,1,1) 
        ed1 = npm.repmat((nIN-1)*nD,nD,1) 
        ed2 = npm.repmat(np.arange(1,nD+1).reshape(nD,1),1,nn)
        ed  = ed1 + ed2                                                         # nodal degrees of freedom
        ed  = ed.flatten('F')                                                   # flatten matrix
        ed  = ed.reshape(nn*nD,1) -1                                            # need -1 for zero indexing    
        fext[ed] = fext[ed].copy() + fp
        
    return fext
def linSolve(np,sp,bc,Kt,oobf,NRit,fd):
    """
    Linear solver
    
    Description:
    -----------
    Function to solve the linear system of equations for the increment in \
displacements and reaction forces.  The linear system is only solved for \
the first Newton-Raphson iteration (NRit>0) onwards as the zeroth \
iteration is required to construct the stiffness matrix based on the \
positions of the material points at the start of the loadstep.  This is \
different from the finite element method where the stiffness matrix from \
the last iteration from the previous loadstep can be used for the zeroth \
iteration. 

    In the case of non-zero displacement boundary conditions, these are \
applied when NRit = 1 and then the displacements for these degrees of \
freedom are fixed for the remaining iterations.

    Parameters
    ----------
    np   : imports numpy module (as np)
    sp   : imports scipy.sparse module
    bc   : boudnary conditions (*,2)
    Kt   : global stiffness matrix (nDoF,nDoF)
    oobf : out of balance force vector (nDoF,1)
    NRit : Newton-Raphson iteration counter (1)
    fd   : free degrees of freedom (*,1)

    Returns
    -------
    duvw : displacement increment (nDoF,1)
    drct : reaction force increment (nDoF,1)
        
    Calling function:
    duvw,drct = linSolve(np,sp,meshData.bc,Kt,oobf,NRit,fd)
    """
    
    import scipy.sparse.linalg as spl
    
    nDoF = oobf.shape[0]                                                        # number of degrees of freedom 
    duvw = np.zeros([nDoF,1])                                                   # zero displacement increment
    drct = np.zeros([nDoF,1])                                                   # zero reaction increment
    
    if NRit > 0:
        bc_idx = bc[:,0]-1                                                      # create non-zero boundary element index
        fd_idx = fd[:,None]                                                     # create free degree of freedom element index
        
        duvw[bc_idx] = (1+np.sign(1-NRit))*np.array(bc[:,1:])                   # apply non-zero boundary conditions
        # need bc[:,1:] to select second column in (20x1) shape, rather 
        # than bc[:,1] otherwise get (20,) which is not compatible
        duvw_fd  = spl.spsolve(Kt[fd_idx,fd], oobf[fd] - Kt[fd_idx,bc_idx] @ duvw[bc_idx]) # solve for displacements
        duvw[fd] = np.reshape(duvw_fd,(len(duvw_fd),1),'f')                     # reshape solved displacements for compatability with displacement array
        drct[bc_idx] = Kt[bc_idx] @ duvw - oobf[bc_idx]                         # determine reaction forces
        
    return duvw,drct
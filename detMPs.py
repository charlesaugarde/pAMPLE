def detMPs(np,npl,npm,sp,uvw,mp_data,nD):
    """
    Stiffness and internal force calculation for all material points
    
    Description:
    -----------
    Function to determine the stiffness contribution of a particle to the \
nodes that it influences based on a Updated Lagrangian finite deformation \
formulation.  The function also returns the stresses at the particles and \
the internal force contribution.  This function allows for elasto-\
plasticity at the material points.  The function is applicable to 1, 2 and \
3 dimensional problems without modification as well as different material \
point methods and background meshes.   

    Parameters
    ----------
    np      : imports numpy module (as np)
    npl     : imports numpy.linalg module (as npl)
    npm     : imports numpy.matlib module (as npm)
    sp      : imports scipy.sparse module (as sp)
    uvw     : nodal displacements that influence the MP (nn*nD,1)
    mp_data : list of objects in mpDataPoint class. Function requires:
                > dSvp  : basis function derivatives (nD,nn)
                > nIN   : background mesh nodes associated with the MP (1,nn)
                > Fn    : previous deformation gradient (3,3) 
                > epsEn : previous elastic logarithmic strain (6,1)
                > mCst  : material constants
                > vp    : material point volume (1)
                > nSMe  : number stiffness matrix entries
    nD      : number of dimensions

    Returns
    -------
    fint    : global internal force vector
    Kt      : global stiffness matrix
    mp_data : list of objects in mpDataPoint class. Function modifies:
                > F     : current deformation gradient (3,3)
                > sig   : current Cauchy stress (6,1)
                > epsE  : current elastic logarithmic strain (6,1)

    Calling function:
    ----------------
    fint,Kt,mp_data = detMPs(np,npl,npm,sp,uvw,mp_data,nD)
    
    See also:
    --------
    formULstiff : updated Lagrangian material stiffness calculation
    Hooke3d     : linear elastic constitutive model
    VMconst     : von Mises elasto-plastic constitutive model
    """
    
    from Hooke3d     import Hooke3d
    from VMconst     import VMconst
    from formULstiff import formULstiff
    
    nmp   = len(mp_data)                                                        # number of material points
    fint1,fint2 = uvw.shape                                                     # get size of internal force vector
    fint  = np.zeros([fint1,fint2])                                             # zero internal force vector
    npCnt = 0                                                                   # counter for the number of entries in Kt
    tnSMe = 0
    for mp in range(0,nmp):                                                     # total number of stiffness matrix entries - iterate through each mp and add 
        tnSMe = tnSMe + mp_data[mp].nSMe
    krow  = np.zeros([tnSMe,1])                                                 # zero the stiffness information
    kcol  = np.zeros([tnSMe,1]) 
    kval  = np.zeros([tnSMe,1]) 
    ddF   = np.zeros([3*3,1])                                                   # create (column) vector instead of matrix (AMPLE) for derivative of duvw wrt. spatial position
    
    # Indexing from MATLAB code used for consistency, 
    # but need to -1 for python zero-indexing in line 34
    if nD == 1:                                                                 # 1D case
        fPos = 0                                                                # deformation gradient positions
        aPos = 0                                                                # material stiffness matrix positions for global stiffness
        sPos = 0                                                                # Cauchy stress components for internal force
    elif nD == 2:                                                               # 2D case (plane strain & stress)
        fPos = [0, 4, 3, 1]
        aPos = [0, 1, 3, 4]
        sPos = [0, 1, 3, 3]
    else:                                                                       # 3D case
        fPos = [0, 4, 8, 3, 1, 7, 5, 2, 6]
        aPos = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        sPos = [0, 1, 2, 3, 3, 4, 4, 5, 5]
    fPos = np.array(fPos); aPos = aPos = np.array(aPos); sPos = np.array(sPos)  # need lists as arrays for indexing
    
    
    for mp in range(0,nmp): 
        nIN = mp_data[mp].nIN                                                   # nodes associated with the material point 
        dNx = mp_data[mp].dSvp                                                  # basis function derivatives (start of lstp)
        nn  = dNx.shape[1]                                                      # no. dimensions & no. nodes
        ed1 = npm.repmat((nIN-1)*nD,nD,1)                                       # 'row vector' (incN) that is copied nD times
        ed2 = np.arange(1,nD+1).reshape(nD,1)
        ed2 = npm.repmat(ed2,1,nn)                                              # nDx1 column vector that is turned into vector (iN long)
        ed  = (ed1 + ed2).flatten(order = 'F') -1                               # degrees of freedom of nodes (vector form)
        
        if nD == 1:                                                             # 1D case
            G = dNx                                                             # strain-displacement matrix
        elif nD == 2:                                                           # 2D case (plane strain & stress) 
            G = np.zeros([4,nD*nn])                                             # zero the strain-disp matrix (2D)
            idxCol1 = np.arange(0,nD*nn,nD)                                     # create indexing for columns
            idxCol2 = np.arange(1,nD*nn,nD)
            G[[[0],[2]], [idxCol1]] = dNx                                       # strain-displacement matrix
            G[[[3],[1]], [idxCol2]] = dNx
        else:                                                                   # 3D case
            G = np.zeros([9,nD*nn])                                             # zero the strain-disp matrix (3D)
            idxCol1 = np.arange(0,nD*nn,nD)                                     # create indexing for columns
            idxCol2 = np.arange(1,nD*nn,nD)
            idxCol3 = np.arange(2,nD*nn,nD)
            G[[[0],[3],[8]], [idxCol1]] = dNx                                   # strain-displacement matrix
            G[[[4],[1],[5]], [idxCol2]] = dNx
            G[[[7],[6],[2]], [idxCol3]] = dNx
        
        ddF = ddF.reshape(9,1)
        ddF[fPos] = G @ uvw[ed]                                                 # spatial gradient (start of lstp) of displacements
        ddF = np.reshape(ddF,(3,3),'F')                                         # reshape to 2D array
        dF  = np.identity(3) + ddF                                              # deformation gradient increment
        F   = dF @ mp_data[mp].Fn                                               # deformation gradient
        
        epsEn  = mp_data[mp].epsEn.copy()                                       # need to make copy to make changes to because assignment also changes original value in list
        epsEn[3:6,0] = 0.5*epsEn[3:6,0]
        epsEn_idx = [0,3,5,3,1,4,5,4,2]
        epsEn = np.reshape(epsEn[epsEn_idx],(3,3),'F')                          # matrix form of previous elastic strain
        De,Ve = npl.eigh(epsEn)                                                 # eigen values and vectors of the elastic strain
        BeT   = dF @ (Ve @ np.diag(np.exp(2*De)) @ Ve.T) @ dF.T                 # trial left Cauchy-Green strain
        # use '@' for matrix multiplication
        Db,Vb = npl.eigh(BeT)                                                   # eigen values and vectors of the trial left Cauchy-Green strain
        
        epsEtr = 0.5*(Vb @ np.diag(np.log(Db)) @ Vb.T).flatten('F')             # trial elastic strain (tensor form)
        epsEtr = epsEtr.reshape(len(epsEtr),1)
        epsEtr_idx = [0,4,8,1,5,2]
        epsEtr = np.diag([1,1,1,2,2,2]) @ epsEtr[epsEtr_idx]                    # trial elastic strain (vector form)
        
        # #--------------------------------------#                              # Constitutive model       
        if mp_data[mp].cmType == 1:
            D, Ksig, epsE = Hooke3d(np,epsEtr,mp_data[mp].mCst)                 # elastic behaviour
        elif mp_data[mp].cmType == 2: 
            D, Ksig, epsE = VMconst(np,npl,epsEtr,mp_data[mp].mCst)             # elasto-plastic behaviour (Von-Mises)
        # #--------------------------------------#
        
        sig  = np.divide(Ksig, npl.det(F))                                      # Cauchy stress
        A    = formULstiff(np,npl,F,D,sig,BeT)                                  # spatial tangent stiffness matrix
        
        iF = npl.solve(dF,np.identity(3))                                       # inverse deformation gradient increment
        iF = iF.flatten('F')  
        dXdx = np.array([[iF[0], 0,     0,     iF[1], 0,     0,     0,     0,     iF[2]],  # start of loadstep to current configuration
                         [0,     iF[4], 0,     0,     iF[3], iF[5], 0,     0,     0    ],  # derivative mapping matrix
                         [0,     0,     iF[8], 0,     0,     0,     iF[7], iF[6], 0    ],     
                         [iF[3], 0,     0,     iF[4], 0,     0,     0,     0,     iF[5]], 
                         [0,     iF[1], 0,     0,     iF[0], iF[2], 0,     0,     0    ],     
                         [0,     iF[7], 0,     0,     iF[6], iF[8], 0,     0,     0    ], 
                         [0,     0,     iF[5], 0,     0,     0,     iF[4], iF[3], 0    ],
                         [0,     0,     iF[2], 0,     0,     0,     iF[1], iF[0], 0    ],
                         [iF[6], 0,     0,     iF[7], 0,     0,     0,     0,     iF[8]]])
        G = dXdx[aPos[:,None],aPos] @ G                                          # derivatives of basis functions (current) 
               
        kp = mp_data[mp].vp*npl.det(dF)*(G.T @ A[aPos[:,None],aPos] @ G)         # material point stiffness contribution        
        fp = mp_data[mp].vp*npl.det(dF)*(G.T @ sig[sPos])                        # internal force contribution
        mp_data[mp].F    = F                                                     # store deformation gradient
        mp_data[mp].sig  = sig                                                   # store Cauchy stress
        mp_data[mp].epsE = epsE                                                  # store elastic logarithmic strain
        
        ed    = np.atleast_2d(ed).astype(int)                                    # need 2nd dimension for shape attribute
        npDoF = ed.shape[0]*(ed.shape[1]**2)                                     # no. entries in kp
        nnDoF = ed.shape[0]*ed.shape[1]                                          # no. DoF in kp
        
        k_idx = np.arange(npCnt,npCnt+npDoF)                                     # index for position storage
        krow[k_idx] = npm.repmat(ed.T,nnDoF,1)                                   # row position storage
        kcol1 = npm.repmat(ed,nnDoF,1).flatten('F')                               
        kcol[k_idx] = np.reshape(kcol1,(len(kcol1),1),'F')                       # column position storage
        kval[k_idx] = np.reshape(kp,(len(kp)**2,1),'F')                          # stiffness storage
        npCnt = npCnt + npDoF                                                    # number of entries in Kt
        fint[ed] = fint[ed].copy() + fp                                          # internal force contribution
    
    nDoF = len(uvw)                                                              # number of degrees of freedom
    # need to flatten storage methods to 1D and convert indexes to integer values
    krow = krow.flatten().astype(int)
    kcol = kcol.flatten().astype(int)
    kval = kval.flatten('F')
    
    # for creating sparse matrix, want 'coo' format, then to manipulate them, 
    # want to convert to another format (eg csr or csc form)
    Kt = sp.coo_matrix((kval, (krow, kcol)),shape = (nDoF,nDoF))                # form the global stiffness matrix
    # choose compressed sparse column matrix for 'faster arithmetic operations,
    # efficient column slicing, fast matrix vector products' 
    Kt = sp.csc_matrix(Kt)

    return fint,Kt,mp_data
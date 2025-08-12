def updateMPs(np,npm,npl,uvw,mp_data):
    """
    Material point update: stress, position and volume

    Description:
    -----------
    Function to update the material point positions and volumes (and domain \
lengths for GIMPM).  The function also updates the previously converged \
value of the deformation gradient and the logarithmic elastic strain at \
each material point based on the converged value and calculates the total \
displacement of each material point.  

    For the generalised interpolation material point method the domain \
lengths are updated according to the stretch tensor following the \
approach of:
Charlton, T.J., Coombs, W.M. & Augarde, C.E. (2017). iGIMP: An implicit \
generalised interpolation material point method for large deformations. \
Computers and Structures 190: 108-125.

    Parameters
    ----------
    np      : imports numpy module (as np)
    npl     : imports numpy.linalg module (as npl)
    npm     : imports numpy.matlib module (as npm)
    sp      : imports scipy.sparse module (as sp)
    uvw     : nodal displacements that influence the MP (nn*nD,1)
    mp_data : list of objects in mpDataPoint class. Function requires:
                  > mpC   : material point coordinates
                  > vp    : material point volume
                  > epsEn : converged elastic strain
                  > Fn    : converged deformation gradient
                  > u     : material point total displacement
                  > lp    : domain lengths (GIMPM only)
                  
    Returns
    -------
    mp_data : list of objects in mpDataPoint class. Function updates \
        attributes listed above
        
    Calling function:
    -----------------
    mp_data  = updateMPs(np,npm,npl,uvw,mp_data)
    """
    
    t = [0,1,2]                                                                 # stretch components for domain updating
    nmp = len(mp_data)                                                          # number of material points
    nD  = len(mp_data[0].mpC)                                                   # choose first point arbitrarily for number of dimensions
        
    for mp in range(0,nmp):
        nIN = mp_data[mp].nIN                                                   # nodes associated with material point    
        nn  = len(nIN)                                                          # number of nodes
        N   = mp_data[mp].Svp                                                   # basis functions        
        F   = mp_data[mp].F                                                     # deformation gradient   
        ed1 = npm.repmat((nIN-1)*nD,nD,1)                                       # 'row vector' (incN) that is copied nD times
        ed2 = np.arange(1,nD+1).reshape(nD,1)
        ed2 = npm.repmat(ed2,1,nn)                                              # nDx1 column vector that is turned into vector (iN long)
        ed  = (ed1 + ed2).flatten(order = 'F') -1                               # nodal degrees of freedom
        uvw_ed = uvw[ed].reshape(nn,2)
        mpU = N @ uvw_ed                                                        # material point displacement    
        
        mp_data[mp].mpC   = mp_data[mp].mpC + mpU                               # update material point coordinates
        mp_data[mp].vp    = npl.det(F)*mp_data[mp].vp0                          # update material point volumes
        mp_data[mp].epsEn = mp_data[mp].epsE                                    # update material point elastic strains        
        mp_data[mp].Fn    = mp_data[mp].F                                       # update material point deformation gradients
        mp_data[mp].u     = mp_data[mp].u + mpU.reshape(len(mpU),1)             # update material point displacements
        
        if mp_data[mp].mpType == 2:                                             # GIMPM only (update domain lengths)      
            D,V = npl.eigh(F.T @ F)                                             # eigen values and vectors F'F
            U   = np.diag(V * np.sqrt(D) @ V.T)                                 # taking diagonal of material stretch matrix
            mp_data[mp].lp = (mp_data[mp].lp0)*U[t[0:nD]]                       # update domain lengths

    return mp_data # outside of for loop

def setupGrid(np):
    """
    Description:
    ------------
    Problem setup for a large deformation elastic beam analysis.
    
    Parameters
    ----------
    np : imports numpy module (as np)

    Returns
    -------
    lstps   : number of loadsteps
    g       : gravity
    mp_data : list of objects in mpDataPoint class with following attributes:
        > mpType : material point type (1 = MPM, 2 = GIMPM)
        > cmType : constitutive model type (1 = elastic, 2 = vM plas.)
        > mpC    : material point coordinates
        > vp     : material point volume
        > vp0    : initial material point volume
        > mpM    : material point mass
        > nIN    : nodes linked to the material point
        > eIN    : element associated with the material point
        > Svp    : basis functions for the material point
        > dSvp   : basis function derivatives (at start of lstp)
        > Fn     : previous deformation gradient
        > F      : deformation gradient
        > sig    : Cauchy stress
        > epsEn  : previous logarithmic elastic strain
        > epsE   : logarithmic elastic strain
        > mCst   : material constants (or internal state parameters)
        > fp     : force at the material point
        > u      : material point displacement
        > lp     : material point domain lengths
        > lp0    : initial material point domain lengths
    meshData : object of mesh class with following attributes:
        > coord : mesh nodal coordinates (nodes,nD)
        > etpl  : element topology (nels,nen)
        > bc    : boundary conditions (*,2)
        > h     : background mesh size (nD,1)

    Calling function
    ----------------
    lstps,g,mp_data,meshData = setupGrid_beam(np)
    
    See also
    ---------
    from2DCoord - background mesh generation
    DetMpPos    - local material point positions
    shapefunc   - background grid basis functions
    """
    
    from formCoord2D import formCoord2D   
    from detMpPos import detMpPos
    from shapefunc import shapefunc
                   
    #%% Analysis parameters
    E      = 12e6                                                               # Young's modulus
    v      = 0.2                                                                # Poisson's ratio   
    fc     = 20e4                                                               # yield strength
    mCst   = [E, v, fc]                                                         # material constants
    g      = 10                                                                 # gravity
    rho    = 0                                                                  # material density
    P      = -5e3                                                               # applied end load
    lstps  = 10                                                                 # number of loadsteps
    a      = 1                                                                  # element multiplier
    nelsx  = 22*a                                                               # number of elements in the x direction
    nelsy  = 20*a                                                               # number of elements in the y direction
    ly     = 10; lx = 11                                                        # domain dimensions
    d = 1; l = 10                                                               # beam dimensions (depth, length)
    mp     = 2                                                                  # number of material points in each direction per element
    mpType = 1                                                                  # material point type: 1 = MPM, 2 = GIMP
    cmType = 1                                                                  # constitutive model: 1 = elastic, 2 = vM plasticity
        
    # Mesh generation
    etpl,coord = formCoord2D(np,nelsx,nelsy,lx,ly)                              # background mesh generation
    nen        = etpl.shape[1]                                                  # number of element nodes. etpl = element topology (one row per element)
    nodes,nD   = coord.shape                                                    # number of nodes and dimensions
    h          = np.array([lx,ly])/np.array([nelsx,nelsy])                      # element lengths in each direction    
    
    #%% Boundary condiitons on background mesh
    bc = np.zeros([(nodes*nD),2])                                               # generate empty bc matrix
    for node in range (1,nodes+1):                                              # loop over nodes
        if coord[node-1, 0] == 0:                                               # roller (x = 0)
            bc[((node*2)-1), :] = np.array([(node*2)-1,0])           
        if coord[node-1, 0] == 0 and coord[node-1, 1] == (ly - d/2):            # mid-depth pin
            bc[(node*2), :] = np.array([node*2,0])  
    bc = bc[~np.all(bc == 0, axis=1)].astype(int)                               # remove empty parts of bc matrix
      
    #%% Define data class 'mesh' to store relevant arrays in
    class mesh:
        def __init__(self,etpl,coord,bc,h,eInA):
            mesh.etpl  = etpl                                                   # element topology (is matrix containing nodes corresponding to elements)
            mesh.coord = coord                                                  # nodal coordinates 
            mesh.bc    = bc                                                     # boundary conditions
            mesh.h     = h                                                      # mesh size
            mesh.eInA  = eInA                                                   # elements in analysis
    
    # storing relevant data structures in mesh class (eInA added later in program where 'int' is)
    meshData = mesh(etpl,coord,bc,h,int)                                        # create meshData object which stores all relevant information (given in brackets)
    
    #%% Material point generation
    ngp    = mp**nD                                                             # number of material points per element
    GpLoc  = detMpPos(np,mp,nD)                                                 # local MP locations (for each element)
    N      = shapefunc(np,nen,GpLoc,nD)                                         # basis functions for the material points
    etplmp,coordmp = formCoord2D(np,20*a,2*a,l,d)                               # mesh for MP generation (last two terms: beam dimensions)                      
    coordmp[:,1]   = coordmp[:,1] + (ly-d)                                      # adjust MP locations (vertical)
    nelsmp = etplmp.shape[0]                                                    # no. elements populated with material points
    nmp    = ngp*nelsmp                                                         # total number of mterial points
    
    mpC = np.zeros([nmp,nD])                                                    # zero MP coordinates
    for nel in range(0,nelsmp):
        indx  = np.arange(nel*ngp+1,(nel+1)*ngp+1)                              # MP locations within mpC
        eC    = coordmp[etplmp[nel]-1,:]                                        # element coordinates. Can't call last element in range (because of zero indexing) so use index-1
        mpPos = N @ eC                                                          # global MP coordinates
        mpC[indx-1,:] = mpPos                                                   # store MP positions
        
    lp = np.zeros([nmp,2])                                                      # zero domain lengths
    lp[:,0] = h[0]/(2*mp)                                                       # domain half length x-direction
    lp[:,1] = h[1]/(2*mp)                                                       # domain half length y-direction                
    vp = 2**nD*lp[:,0]*lp[:,1]                                                  # volume associated with each material point
    vp = np.reshape(vp,(len(vp),1))
       
    #%% defining class with necessary attributes each material point should have 
    class mpDataPoint:
        def __init__(self,mpType,cmType,mpC,vp,vp0,mpM,
                     nIN,eIN,Svp,dSvp,Fn,F,sig,
                     epsEn,epsE,mCst,fp,u,lp,lp0):
            self.mpType = mpType                                                # material point type: 1 = MPM, 2 = GIMP
            self.cmType = cmType                                                # constitutive model: 1 = elastic, 2 = vM plasticity
            self.mpC    = mpC                                                   # material point coordinates 
            self.vp     = vp                                                    # material point volume                    
            self.vp0    = vp0                                                   # material point initial volume
            self.mpM    = mpM                                                   # material point mass
            self.nIN    = nIN                                                   # nodes associated with the material point
            self.eIN    = eIN                                                   # element associated with the material point
            self.Svp    = Svp                                                   # material point basis functions
            self.dSvp   = dSvp                                                  # derivative of the basis functions
            self.Fn     = Fn                                                    # previous deformation gradient
            self.F      = F                                                     # deformation gradient
            self.sig    = sig                                                   # Cauchy stress
            self.epsEn  = epsEn                                                 # previous elastic strain (logarithmic)
            self.epsE   = epsE                                                  # elastic strain (logarithmic)
            self.mCst   = mCst                                                  # material constants (or internal variables) for constitutive model
            self.fp     = fp                                                    # point forces at material points
            self.u      = u                                                     # material point displacements
            self.lp     = lp                                                    # material point domain lengths
            self.lp0    = lp0                                                   # initial material point domain lengths
    
    #%% Creating list list of material points
    
    # Defines empty list to store material point objects
    mp_data = []
    
    # Creates individual material points with necessary attributes
    # in for loop and adds them to mp_data list: 
    # eg. mp_data[0].mpC calls mpC attribute from first element in list
    for mp in range(0,nmp):
        if abs(mpC[mp,0] - (l-lp[mp,0])) < 1e-3/a and abs(mpC[mp,1] - (ly-d/2)) < (lp[mp,1] + 1e-3/a):
            fp = np.array([[0],[P/2]])                                          # point forces at material points (end load)
        else:
            fp = np.zeros([nD,1])                                               # point forces at material points
        
        mp_point = mpDataPoint(mpType,               # mpType
                               cmType,               # cmType
                               mpC[mp],              # mpC
                               vp[mp],               # vp
                               vp[mp],               # vp0
                               vp[mp]*rho,           # mpM
                               np.zeros([nen,1]),    # nIN
                               0,                    # eIN
                               np.zeros([1,nen]),    # Svp
                               np.zeros([nD,nen]),   # dSvp
                               np.identity(3),       # Fn
                               np.identity(3),       # F
                               np.zeros([6,1]),      # sig
                               np.zeros([6,1]),      # epsEn
                               np.zeros([6,1]),      # epsE
                               mCst,                 # mCst
                               fp,                   # fp
                               np.zeros([nD,1]),     # u
                               np.zeros([1,nD]),     # lp  - material point domain lengths (for MPM). Redefined for GIMP below
                               np.zeros([1,nD]))     # lp0 - initial material point domain lengths (MPM). Redefined for GIMP below
        
        if mp_point.mpType == 2:
            mp_point.lp  = lp[mp,:]                                             # material point domain lengths (GIMP)
            mp_point.lp0 = lp[mp,:]                                             # initial material point domain lengths (GIMP)
        
        # add material point object to mp_data list
        mp_data.append(mp_point)

    return lstps,g,mp_data,meshData
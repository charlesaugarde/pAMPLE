def makeVtkMP(pyvtk,np,mpC,sig,uvw,mpFileName):
    """
    VTK output file generation: material point data
    
    Description
    -----------
    Function to generate a VTK file containing the material point data.
    
    Parameters
    ----------
    mpC        : material point coordinates (nmp,nD)
    sig        : material point stresses (nmp,6)
    uvw        : material point displacements (nmp,nD)
    mpFileName : VTK file name, for example 'mpData.vtk'  
    
    Calling function:
    -----------------
    makeVtkMP(pyvtk,np,mpC,sig,uvw,mpFileName)
    """  
    
    from pyvtk import VtkData, PointData,Scalars
    from AMPLE_unstructuredGrid import UnstructuredGrid                         # import adapted UnstructuredGrid class
    
    nmp,nD = mpC.shape                                                          # number of material pints and dimensions
    
    # Position output
    if nD < 3:
        mpC = np.concatenate((mpC, np.zeros([nmp,3-nD])),axis = 1)
    mpC = list(tuple(map(tuple,mpC)))                                           # List the nodal co-ordinates for the VtkData input
    
    structure = UnstructuredGrid(mpC)
    
    # Stress (sig) and displacement (uvw) outputs 
    if nD == 3:                                                                 # 3D case            
        pointData = PointData(\
            Scalars(sig[:,0],name='sigma_xx'),
            Scalars(sig[:,1],name='sigma_yy'),
            Scalars(sig[:,2],name='sigma_zz'),
            Scalars(sig[:,3],name='sigma_xy'),
            Scalars(sig[:,4],name='sigma_yz'),
            Scalars(sig[:,5],name='sigma_zx'),
            Scalars(uvw[:,0],name='u_x'),
            Scalars(uvw[:,1],name='u_y'),
            Scalars(uvw[:,2],name='u_z'))                                       

    elif nD == 2:                                                               # 2D    
        pointData = PointData(\
            Scalars(sig[:,0],name='sigma_xx'),
            Scalars(sig[:,1],name='sigma_yy'),
            Scalars(sig[:,2],name='sigma_zz'),
            Scalars(sig[:,3],name='sigma_xy'),
            Scalars(sig[:,4],name='sigma_yz'),
            Scalars(sig[:,5],name='sigma_zx'),
            Scalars(uvw[:,0],name='u_x'),
            Scalars(uvw[:,1],name='u_y'))
    
    elif nD == 1:                                                               # 1D
        pointData = PointData(\
            Scalars(sig[:,0],name='sigma_xx'),
            Scalars(sig[:,1],name='sigma_yy'),
            Scalars(sig[:,2],name='sigma_zz'),
            Scalars(sig[:,3],name='sigma_xy'),
            Scalars(sig[:,4],name='sigma_yz'),
            Scalars(sig[:,5],name='sigma_zx'),
            Scalars(uvw[:,0],name='u_x'))
       
    vtk = VtkData(structure,pointData,'Python generated vtk file, EJB')         # generate vtk data
    vtk.tofile(mpFileName)                                                      # create vtk file
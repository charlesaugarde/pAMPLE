def postPro(os,np,pyvtk,cwd,mp_data,lstp,meshData):
    """
    Post processing script for the AMPLE code

    Description:
    -----------
    The script produces VTK output files based on the background mesh and \
material point data. Background mesh is plotted for lstp = 0.  

    Parameters
    ----------
    os       : imports operating system module (as os)
    np       : imports numpy module as (np)
    pyvtk    : imports pyvtk module
    cwd      : current working directory address
    mp_data  : list of objects of mpDataPoint class
    lstp     : current loadstep
    meshData : mesh class object

    Returns
    -------
    None.

    Calling function:
    ----------------
    postPro(os,np,pyvtk,cwd,mp_data,lstp,meshData)
    
    See also
    --------
    makeVtk           - VTK file for background mesh
    makeVtkMP         - VTK file for MP data
    """

    from makeVtkMP import makeVtkMP
    from makeVtk   import makeVtk
    
    # Create path and folder to store vtk files in
    output_files = r'Output files'
    if not os.path.exists(output_files):
        os.makedirs(output_files)
    
    # directory location and string format for vtk output file name
    mpDataName = os.path.join(cwd,output_files,f'mpData_{lstp}.vtk')
    nmp = len(mp_data)
    nD  = meshData.coord.shape[1]
    sig = np.zeros([nmp,6])                                                     # initialise all material point data for vtk file plot                      
    mpC = np.zeros([nmp,nD])
    mpU = np.zeros([nmp,nD])
    
    # loop through mp_data list to add all sig, mpC and mpU attributes from 
    # each data point to array to add to input to vtk file
    for mp in range(0,nmp):
        sig[mp,:] = mp_data[mp].sig.reshape(1,6)                                # all material point stresses (nmp,6)
        mpC[mp,:] = mp_data[mp].mpC.reshape(1,nD)                               # all material point coordinates (nmp,nD)
        mpU[mp,:] = mp_data[mp].u.reshape(1,nD)                                 # all material point displacements
        
    makeVtkMP(pyvtk,np,mpC,sig,mpU,mpDataName)                                  # generate material point VTK file
    
    meshName = os.path.join(cwd,output_files,'mesh.vtk')                        # generate mesh VTK filename
    if lstp == 0:
        makeVtk(np,pyvtk,meshData.coord,meshData.etpl,meshName)                 # generate mesh VTK file
def makeVtk(np,pyvtk,coord,etpl,meshName):
    """
    VTK output file generation: mesh data

    Author: Eleanor Bacon
    Date:   21/01/2021
    
    Description
    -----------
    Function to generate a VTK file containing the background mesh data.
    Requires pyVTK package: can be installed via:
        >  pip install pyvtk

    Parameters
    ----------
    coord    : coordinates of the grid nodes (nodes,nD)
    etpl     : element topology (nels,nen) 
    meshName : VTK file name, for example 'mesh.vtk'  
    
    Returns
    -------
    None.
    
    Calling function:
    ----------------
    makeVtk(np,pyvtk,meshData.coord,meshData.etpl,meshName) 
    """
    
    from pyvtk import VtkData
    from AMPLE_unstructuredGrid import UnstructuredGrid
    # pyVtk.UnstructuredGrid adapted by EB to add in quadratic hexahedral  
    # and quadratic quadrilateral VTK elements
    
    nodes, nD = coord.shape
    nels, nen = etpl.shape
    
    # nodal coordinates
    if nD < 3:
        coord = np.concatenate((coord, np.zeros([nodes,3-nD])),axis = 1)
    coord_nodal = list(tuple(map(tuple,coord))) # List the nodal co-ordinates for the VtkData input
  
    # FEM element topology to VTK format
    # element topology - ordering elements in ascending order with zero-indexing: 
    #   > etpl_vtk = etpl[:,tvtk]-1   
    if nD == 3:                                                                 # 3D case     
        if nen == 20:
            tvtk = [0,6,18,12,2,4,16,14,7,11,19,8,3,10,15,9,1,5,17,13]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,quadratic_hexahedron = etpl_vtk)
        elif nen == 8:
            tvtk = [0,3,7,4,1,2,6,5]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,hexahedron = etpl_vtk)
        elif nen == 10:
            tvtk = [0,1,2,3,4,5,6,7,9,8]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,quadratic_tetra = etpl_vtk)
        elif nen == 4:
            tvtk = [0,2,1,3]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,tetra = etpl_vtk)
        elif nen == 9:
            tvtk = [2,0,6,4,1,7,5,3,8]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,tetra = etpl_vtk)
    
    elif nD == 2:                                                               # 2D case
        if nen == 3:
            tvtk = [0,2,1]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,triangle = etpl_vtk)
        elif nen == 4:
            tvtk = [0,3,1,2]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,pixel = etpl_vtk)
        elif nen == 8:
            tvtk = [0,6,4,2,7,5,3,1]
            etpl_vtk = etpl[:,tvtk]-1
            structure = UnstructuredGrid(coord_nodal,quadratic_quad = etpl_vtk)
    
    # use pyVTK module VtkData to generate VTK data file
    vtk = VtkData(structure,'Python generated vtk file, EJB')
    vtk.tofile(meshName)
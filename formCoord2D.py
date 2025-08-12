def formCoord2D(np,nelsx,nelsy,lx,ly):
    """
    Two dimensional finite element grid generation
    
    Description:
    ------------
    Function to generate a 2D finite element grid of linear quadrilateral elements.

    Parameters
    ----------
    np    : imports numpy module (as np)
    nelsx : number of elements in the x direction
    nelsy : number of elements in the y direction
    lx    : length in the x direction
    ly    : length in the y direction    

    Returns
    -------
    etpl  : element topology
    coord : nodal coordinates
    
    Calling function:
    ----------------
    etpl,coord = formCoord2D(np,nelsx,nelsy,lx,ly)
    """
    
    nels  = nelsx*nelsy                                                         # number of elements
    nodes = (nelsx+1)*(nelsy+1)                                                 # number of nodes
    
    # node generation
    coord = np.zeros([nodes,2])                                                 # zero coordinates
    node  = 0                                                                   # zero node counter
    for j in range(0,(nelsy+1)):
        y = (ly*j)/nelsy
        for i in range(0,nelsx+1):
            x = (lx*i)/nelsx
            coord[node,:] = np.array([x, y])
            node = node + 1
    
    # element generation
    etpl = np.zeros([nels,4])                                                   # initialise etpl matrix of zeros (no. elements)x(no. nodes)
    nel  = 0                                                                    # zero element counter
    for nely in range(1,nelsy+1):
        for nelx in range(1,nelsx+1):
            etpl[nel][0] = ((nely-1)*(nelsx+1)) + nelx
            etpl[nel][1] = (nely*(nelsx+1)) + nelx
            etpl[nel][2] = (etpl[nel][1]) + 1
            etpl[nel][3] = (etpl[nel][0]) + 1
            nel = nel + 1  
    
    etpl = etpl.astype(int)                                                     # convert etpl matrix to integer values (so can use values for indexing later)
       
    return etpl,coord
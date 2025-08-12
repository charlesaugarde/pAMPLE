def nodesForMP(np,etpl,elems):  #elems = eIN
    """
    Unique list of nodes associated with a material point

    Description:
    ------------
    Function to determine a unique list of nodes for a group of elements \
connected to a material point.
        
    Parameters
    ----------
    np    : import numPy as np
    etpl  : element topology (nels,nen)
    elems : elements in the group (n,1)

    Returns
    -------
    nodes : list containing the nodes associated with the elements
    
    Calling function:
    ----------------
    nodes = nodesForMP(np,etpl,elems)
    """

    e1  = etpl[elems,:].flatten('F')                                            # list of all nodes (inc. duplicates)  
    e   = np.sort(e1).tolist()                                                  # sorting list of nodes in ascending order
    nodes_list = list(dict.fromkeys(e))                                         # unique list of nodes
    nodes = np.asarray(nodes_list)
    
    return nodes
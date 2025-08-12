def SvpMPM(xp,xv,h):
    """
    1D material point basis functions
    
    Description:
    ------------
    Function to determine the one dimensional MPM shape functions based on\
global coordinates.
        
    Parameters
    ----------
    xp : particle position
    xv : grid node position
    h  : element length
    
    Returns
    -------
    Svp  : particle characteristic function
    dSvp : gradient of the characterstic function 
    
    Calling function:
    ----------------
    Svp,dSvp = SVPMPM(xp,xv,h)    
    """
    if -h < (xp - xv) and (xp - xv) <= 0:       # MP in 'left' element
        Svp  = 1 + (xp-xv)/h
        dSvp = 1/h
    
    elif 0 < (xp - xv) and (xp - xv) <= h:      # MP in 'right' element
        Svp  = 1 - (xp-xv)/h
        dSvp = -1/h
    
    else:                                       # MP outside of element
        Svp  = 0
        dSvp = 0
    
    return Svp, dSvp
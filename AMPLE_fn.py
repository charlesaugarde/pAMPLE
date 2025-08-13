def main():
    ## IMPORT STANDARD LIBRARIES
    import numpy as np
    import numpy.linalg as npl
    import numpy.matlib as npm
    import scipy.sparse as sp
    import os
    import time
    import pyvtk 
    
    # pyVTK must be installed in Python environment before running AMPLE
    #   > downloadable from PyPI using: pip install PyVTK
    
    ## IMPORT LOCAL APPLICATIONS
    # from setupGrid   import setupGrid
    from setupGrid_beam       import setupGrid
    # from setupGrid_beam_Jupyter       import setupGrid    
    # from setupGrid_collapse   import setupGrid
    from elemMPinfo  import elemMPinfo
    from detExtForce import detExtForce 
    from detFDoFs    import detFDoFs
    from linSolve    import linSolve
    from detMPs      import detMPs
    from updateMPs   import updateMPs
    from postPro     import postPro
    
    # Start timer
    tic = time.perf_counter()
    
    # Get current working directory 
    cwd = os.getcwd() 
    
    NRitMax = 10; tol = 1e-9
    lstps,g,mp_data,meshData = setupGrid(np) 
    nodes, nD = meshData.coord.shape
    nels,nen  = meshData.etpl.shape
    nDoF = nodes*nD
    nmp = len(mp_data)
    lstp = 0
    
    postPro(os,np,pyvtk,cwd,mp_data,lstp,meshData)
    for lstp in range(1,lstps+1):    
        print(f'\nLoadstep {lstp} of {lstps}')
        meshData,mp_data = elemMPinfo(np,mp_data,meshData)
        fext = detExtForce(np,npm,nodes,nD,g,mp_data)
        fext = fext*lstp/lstps
        oobf = fext
        fErr = 1
        frct = np.zeros([nDoF,1])
        uvw  = np.zeros([nDoF,1])
        fd   = detFDoFs(np,npm,meshData)-1
        NRit = 0
        Kt   = 0
        
        while (fErr > tol) and (NRit < NRitMax) or (NRit < 2):
            duvw,drct = linSolve(np,sp,meshData.bc,Kt,oobf,NRit,fd)
            uvw  = uvw + duvw
            frct = frct + drct
            fint,Kt,mp_data = detMPs(np,npl,npm,sp,uvw,mp_data,nD)
            eps  = np.finfo(float).eps
            oobf = fext - fint + frct
            fErr = npl.norm(oobf)/npl.norm(fext + frct + eps)
            NRit = NRit+1
            print(f'\tIteration {NRit}      NR error {fErr:.3e}')
        
        # updating MPs and postPro vtk files is still inside for-loop
        mp_data  = updateMPs(np,npm,npl,uvw,mp_data)
        postPro(os,np,pyvtk,cwd,mp_data,lstp,meshData)
        
    # Exit loop and stop timer    
    toc = time.perf_counter()
    
    print('Program finished - thank you for waiting')
    print(f"Completed program in {toc-tic:0.3f} seconds")
    run_time = toc-tic
    
    return run_time
"""
AMPLE-Py: A Material Point Learning Environment in Python

Author: William Coombs (adapted from MATLAB to Python by Eleanor Bacon)
Date:   28/02/2021

Description: 
    elasto-plastic (EP) material point method (MPM) code based on an 
    updated Lagrangian (UL) description of motion with a quadrilateral 
    background mesh.

Functions called:
    SETUPGRID   - analysis specific information
    ELEMMPINFO  - material point-element information
    DETEXTFORCE - external forces
    DETFDOFS    - mesh unknown degrees of freedom
    LINSOLVE    - linear solver
    DETMPS      - material point stiffness and internal force
    UPDATEMPS   - update material points
    POSTPRO     - post processing function including vtk output    
    
Note:
    pyVTK must be installed in Python environment before running AMPLE
        > downloadable from PyPI using: pip install PyVTK
"""

## IMPORT STANDARD LIBRARIES
import numpy as np
import numpy.linalg as npl
import numpy.matlib as npm
import scipy.sparse as sp
import os
import time
import pyvtk 

## IMPORT LOCAL APPLICATIONS
## Setup function choices:                                                      # analysis specific information 
# from setupGrid              import setupGrid                                    # (column under self-weight)
from setupGrid_beam         import setupGrid                                    # (elastic cantilever beam)
# from setupGrid_collapse     import setupGrid                                    # (elasto-plastic collapse)
# from setupGrid_beam_Jupyter import setupGrid                                    # (simplified elastic cantilever beam)

## Standard functions
from elemMPinfo  import elemMPinfo                                              # material point-element information
from detExtForce import detExtForce                                             # external forces
from detFDoFs    import detFDoFs                                                # mesh unknown degrees of freedom
from linSolve    import linSolve                                                # linear solver
from detMPs      import detMPs                                                  # material point stiffness and internal force
from updateMPs   import updateMPs                                               # update material points
from postPro     import postPro                                                 # post processing function including vtk output  

tic = time.perf_counter()                                                       # start timer
cwd = os.getcwd()                                                               # get current working directory 

lstps,g,mp_data,meshData = setupGrid(np)                                        # setup information
NRitMax = 10; tol = 1e-9                                                        # Newton Raphson parameters
nodes, nD = meshData.coord.shape                                                # number of nodes and dimensions    
nels,nen  = meshData.etpl.shape                                                 # number of elements and nodes/element
nDoF = nodes*nD                                                                 # total number of degrees of freedom
nmp  = len(mp_data)                                                             # number of material points
lstp = 0                                                                        # zero loadstep counter (for plotting function)
postPro(os,np,pyvtk,cwd,mp_data,lstp,meshData)                                  # plotting initial state & mesh

for lstp in range(1,lstps+1):                                                   # loadstep loop
    print(f'\nLoadstep {lstp} of {lstps}')                                      # text output to screen (loadstep)
    meshData,mp_data = elemMPinfo(np,mp_data,meshData)                          # material point - element information 
    fext = detExtForce(np,npm,nodes,nD,g,mp_data)                               # external force calculation (total)
    fext = fext*lstp/lstps                                                      # current external force value
    oobf = fext                                                                 # initial out-of-balance force
    fErr = 1                                                                    # initial error
    frct = np.zeros([nDoF,1])                                                   # zero the reaction forces
    uvw  = np.zeros([nDoF,1])                                                   # zero the displacements
    fd   = detFDoFs(np,npm,meshData)-1                                          # free degrees of freedom
    NRit = 0                                                                    # zero the iteration counter
    Kt   = 0                                                                    # zero global stiffness matrix
    
    while (fErr > tol) and (NRit < NRitMax) or (NRit < 2):                      # global equilibrium loop
        duvw,drct = linSolve(np,sp,meshData.bc,Kt,oobf,NRit,fd)                 # linear solver
        uvw  = uvw  + duvw                                                       # update displacements
        frct = frct + drct                                                      # update reaction forces
        fint,Kt,mp_data = detMPs(np,npl,npm,sp,uvw,mp_data,nD)                  # global stiffness & internal force
        eps  = np.finfo(float).eps                                              # floating-point relative accuracy
        oobf = fext - fint + frct                                               # out-of-balance force (oobf)
        fErr = npl.norm(oobf)/npl.norm(fext + frct + eps)                       # normalised oobf error
        NRit = NRit+1                                                           # increment the NR counter    
        print(f'\tIteration {NRit}      NR error {fErr:.3e}')                   # text output to screen
    
    # updating MPs and postPro vtk files is still inside for-loop
    mp_data  = updateMPs(np,npm,npl,uvw,mp_data)                                # update material points            
    postPro(os,np,pyvtk,cwd,mp_data,lstp,meshData)                              # plotting and post-processing            
       
toc = time.perf_counter()                                                       # exit for loop and stop timer

print('Program finished - thank you for waiting.')
print(f"Completed program in {toc-tic:0.3f} seconds.")
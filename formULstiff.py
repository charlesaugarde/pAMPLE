def formULstiff(np,npl,F,D,s,B):
    """
    Updated Lagrangian material stiffness matrix
    
    Description
    -----------
    Function to determine consistent material stiffness matrix based on an \
updated Lagrangian formulation of finite deformation mechanics.  See \
equations (25) and (26) of the following paper for full details: \
Charlton, T.J., Coombs, W.M. & Augarde, C.E. (2017). iGIMP: An implicit \
generalised interpolation material point method for large deformations. \
Computers and Structures 190: 108-125.

    Parameters
    ----------
    np  : imports numpy module (as np)
    npl : imports numpy.linalg module (as npl)
    F   : deformation gradient (3,3)
    D   : small strain material stiffness matrix (6,6)
    s   : Cauchy stress (6,1)
    B   : trial elastic left Cauchy-Green strain matrix (3,3)

    Returns
    -------
    A : consistent tangent stiffness matrix (9,9)
    
    Calling function:
    -----------------
    A = formULstiff(np,npl,F,D,s,B)
    
    See also:
    ---------
    parDerGen : partial derivative of a second order tensor
    """
    
    t = np.array([0, 1, 2, 3, 3, 4, 4, 5, 5])                                   # 6 to 9 component steering vector
    J = npl.det(F)                                                              # volume ratio
    bP,bV = npl.eigh(B)                                                         # eigen values/vector of the trial elastic strain tensor
    bP = bP.reshape(3,1)                                        
    L  = parDerGen(np,npl,B,bV,bP,np.log(bP),1/bP)                              # derivative of the logarithmic strain
    
    s = s.flatten('F')                                                          # flatten matrix so can use element-wise referencing as in MATLAB AMPLE
    # S is the matrix form of sigma_{il}delta_{jk}
    S = np.array([[s[0], 0,    0,    s[3], 0,    0,    0,    0,    s[5]],       
                  [0,    s[1], 0,    0,    s[3], s[4], 0,    0,    0   ],
                  [0,    0,    s[2], 0,    0,    0,    s[4], s[5], 0   ],
                  [0,    s[3], 0,    0,    s[0], s[5], 0,    0,    0   ],
                  [s[3], 0,    0,    s[1], 0,    0,    0,    0,    s[4]],
                  [0,    0,    s[4], 0,    0,    0,    s[1], s[3], 0   ],
                  [0,    s[4], 0,    0,    s[5], s[2], 0,    0,    0   ],
                  [s[5], 0,    0,    s[4], 0,    0,    0,    0,    s[2]],
                  [0,    0,    s[5], 0,    0,    0,    s[3], s[0], 0  ]])
    
    B = B.flatten('F')                                                          # trial elastic left Cauchy-Green strain matrix

    # T is matrix form of delta_{pk}b^e_{ql}+delta_{qk}b^e_{pl}
    T = np.array([[2*B[0], 0,      0,      2*B[3], 0,       0,      2*B[6], 0,      0   ],          
                  [0,      2*B[4], 0,      0,      2*B[1],  2*B[7], 0,      0,      0   ],
                  [0,      0,      2*B[8], 0,      0,       0,      2*B[5], 2*B[2], 0   ],
                  [B[1],   B[3],   0,      B[4],   B[0],    B[6],   0,      0,      B[7]],
                  [B[1],   B[3],   0,      B[4],   B[0],    B[6],   0,      0,      B[7]],
                  [0,      B[5],   B[7],   0,      B[2],    B[8],   B[4],   B[1],   0   ],
                  [0,      B[5],   B[7],   0,      B[2],    B[8],   B[4],   B[1],   0   ],
                  [B[2],   0,      B[6],   B[5],   0,       0,      B[3],   B[0],   B[8]],
                  [B[2],   0,      B[6],   B[5],   0,       0,      B[3],   B[0],   B[8]]])
    
    D_t = D[t[:,None],t]
    L_t = L[t[:,None],t]
    A = D_t @ L_t @ T / (2*J)-S                                                 # consistent tangent stiffness matrix
    return A

def parDerGen(np,npl,X,eV,eP,yP,ydash):
    """
    Description
    -----------
    Description:
Function to determine the partial derivative of a second order tensor \
function with respect to its argument (X) based on the implementation \
described by in the following paper:

C. Miehe, Comparison of two algorithms for the computation of fourth-\
order isotropic tensor functions, Computers & Structures 66 (1998) 37-43.

For example, in order to determine the derivative of log(X) with respect \
to X the inputs to the function should be:

[L] = parDerGen(X,eV,eP,log(eP),1./eP)
as the derivative of the log(x) is 1/x

The symbols used in the code follow, as closely as possible, those used \
in the Miehe (1998) paper.  There are a number of different cases that \
have to be checked (all zero and repeated eigenvalues) in addition to the \
general case where there are no repeated eigenvalues.  
    
    Parameters
    ----------
    np    : imports numpy module (as np)
    npl   : imports numpy.linalg module (as npl)
    X     : second order tensor in matrix format (3,3)
    eV    : eigenvectors of X (3,3)
    eP    : eigenvalues of X (1,3) 
    yP    : function applied to eP (1,3)
    ydash : derivative of the function applied to eP (1,3)

    Returns
    -------
    L : partial derivative of the second order tensor with respect to its \
argument (6,6)
    
    Calling function:
    ----------------
    L = parDerGen(np,npl,X,eV,eP,yP,ydash)
    """
    
    Xmat = X                                                                    # saving array
    X = X.flatten('F')                                                          # flatten X so can use element-wise indexing
    tol = 1e-9
    Is = np.diag([1, 1, 1, 0.5, 0.5, 0.5]) 
    if abs(eP[0])<tol and abs(eP[1])<tol and abs(eP[2])<tol:                    # all zero eigenvalues case
        L = Is
    elif abs(eP[0]-eP[1])<tol and abs(eP[0]-eP[2])<tol:                         # equal eigenvalues case
        L = ydash[0]*Is # have this case atm
    elif abs(eP[0]-eP[1])<tol or abs(eP[1]-eP[2])<tol or abs(eP[0]-eP[2])<tol:  # repeated eigenvalues case
        if abs(eP[0]-eP[1]) < tol:
            xa  = eP[2];    xc  = eP[0]
            ya  = yP[2];    yc  = yP[0]
            yda = ydash[2]; ydc = ydash[0]
        elif abs(eP[1]-eP[2]) < tol:
            xa  = eP[0];    xc  = eP[1]
            ya  = yP[0];    yc  = yP[1]
            yda = ydash[0]; ydc = ydash[1]
        else:
            xa  = eP[1];    xc  = eP[0]
            ya  = yP[1];    yc  = yP[0]
            yda = ydash[1]; ydc = ydash[0]
            
        X_idx = [0,4,8,3,5,2]
        x  = np.reshape(X[X_idx],(6,1),'F')
        s1 = (ya-yc)/(xa-xc)**2-ydc/(xa-xc)
        s2 = 2*xc*(ya-yc)/(xa-xc)**2-(xa+xc)/(xa-xc)*ydc
        s3 = 2*(ya-yc)/(xa-xc)**3-(yda+ydc)/(xa-xc)**2
        s4 = xc*s3
        s5 = (xc**2)*s3
        
        # X flattened above so can use element-wise referencing
        r1 = [2*X[0], 0,      0,      X[1],           0,              X[2]         ]
        r2 = [0,      2*X[4], 0,      X[1],           X[5],           0            ]
        r3 = [0,      0,      2*X[8], 0,              X[5],           X[2]         ]
        r4 = [X[1],   X[1],   0,      (X[0]+X[4])/2,  X[2]/2,         X[5]/2       ]
        r5 = [0,      X[5],   X[5],   X[2]/2,         (X[4]+X[8])/2,  X[1]/2       ]
        r6 = [X[2],   0,      X[2],   X[5]/2,         X[1]/2,         (X[0]+X[8])/2]
        dX2dX = np.concatenate((r1,r2,r3,r4,r5,r6),axis = 0).reshape(6,6)
        
        bm1  = np.array([1,1,1,0,0,0]).reshape(6,1)
        bm11 = np.array([[1,1,1,0,0,0],
                         [1,1,1,0,0,0],
                         [1,1,1,0,0,0],
                         [0,0,0,0,0,0],
                         [0,0,0,0,0,0],
                         [0,0,0,0,0,0]])
        L = (s1*dX2dX) - (s2*Is) - (s3*(x*x.T)) + (s4*(x@(bm1.T) + bm1@(x.T))) - (s5*bm11)
    else:                                                                       # general case (no repeated eigenvalues)        
        D = np.array([(eP[0]-eP[1])*(eP[0]-eP[2]),
                      (eP[1]-eP[0])*(eP[1]-eP[2]),
                      (eP[2]-eP[0])*(eP[2]-eP[1])]).reshape(3,1)
        alfa=0; bta=0; gama=np.zeros([3,1]); eDir=np.zeros([6,3])
        for i in range(0,3):
            alfa = alfa + yP[i]*eP[i]/D[i]
            bta  = bta + yP[i]/D[i]*npl.det(Xmat)
            for j in range(0,3):
                gama[i] = gama[i] + yP[j]*eP[j]/D[j]*(npl.det(Xmat)/eP[j]-eP[i]**2)*1/eP[i]**2
            esq = eV[:,i].reshape(3,1) @ eV[:,i].reshape(1,3)
            eDir[:,i] = np.array([esq[0,0],esq[1,1],esq[2,2],esq[0,1],esq[1,2],esq[2,0]])
        
        y = npl.inv(Xmat).flatten('F')
        Ib_r1 = [y[0]**2,   y[1]**2,    y[6]**2,     y[0]*y[1],               y[1]*y[6],                y[0]*y[6]]
        Ib_r2 = [y[1]**2,   y[4]**2,    y[5]**2,     y[4]*y[1],               y[4]*y[5],                y[1]*y[5]]
        Ib_r3 = [y[6]**2,   y[5]**2,    y[8]**2,     y[5]*y[6],               y[8]*y[5],                y[8]*y[6]]
        Ib_r4 = [y[0]*y[1], y[4]*y[1],  y[5]*y[6],  (y[0]*y[4]+y[1]**2)/2,   (y[1]*y[5]+y[4]*y[6])/2,  (y[0]*y[5]+y[1]*y[6])/2]
        Ib_r5 = [y[1]*y[6], y[4]*y[5],  y[8]*y[5],  (y[1]*y[5]+y[4]*y[6])/2, (y[8]*y[4]+y[5]**2)/2,    (y[8]*y[1]+y[5]*y[6])/2]
        Ib_r6 = [y[0]*y[6], y[1]*y[5],  y[8]*y[6],  (y[0]*y[5]+y[1]*y[6])/2, (y[8]*y[1]+y[5]*y[6])/2,  (y[8]*y[0]+y[6]**2)/2  ]
        
        Ib = np.concatenate((Ib_r1,Ib_r2,Ib_r3,Ib_r4,Ib_r5,Ib_r6),axis = 0).reshape(6,6)
        L = alfa*Is - bta*Ib
        for i in range(0,3):
            L = L + (ydash[i] + gama[i])*eDir[:,i]*eDir[:,i].reshape(6,1)
    return L
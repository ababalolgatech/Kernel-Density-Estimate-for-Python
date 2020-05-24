"""
@author: Dr. Ayodeji Babalola
"""

import math 
import numpy as np
import BallTreeDens_module 
import DualTree_module
import BallTree_module
import knn_module 
from statsmodels.stats.weightstats import DescrStatsW
from scipy.special import gamma

#----------------------------------------------------
def ad(a,b,d= None):
    if (d is None):
        c = {a,b}
    else:
        c = {a,b,d}    
    return c
#-----------------------------------------------------------------
def matlab_any(arr):
    if (arr.size == 0):
        out = False
    else:
        tmp = np.asarray(arr.nonzero())
        tmp = tmp.sum()
        
        if(tmp ==0):
            out = True
        else:
            out = False    
    return out
#-----------------------------------------------------------------
def var(arr,axis_=0):
    out = np.var(arr,axis = axis_,ddof=1)
    # ddof is degree of freedom  
    return out
#-----------------------------------------------------------------
def std(arr,axis_=0):
    out = np.std(arr,axis = axis_,ddof=1)
    # ddof is degree of freedom     
    return out
#-----------------------------------------------------------------
def m_size(arr):
    
    if(type(arr) == tuple):
     sz = (1,1)   
    else:
        try:
            sz =( arr.shape[0],arr.shape[1])
        except :
            try:
                sz =( 1,arr.shape[1])
            except:
               try :  
                   sz =( arr.shape[0],1)
               except:
                   try:
                       tmp = arr.size
                       if (tmp==1):
                           sz = (1,1)
                       else:
                           sz=(tmp,1)
                   except :
                        pass               
    return sz
#-----------------------------------------------------------------
def extr_vec(arr,row,col):
    if(type(row)==list or type(col) == list or type(row) == np.ndarray):
        # 2D
        if (len(row) == 2):
            tmp = arr[row[0]:row[1]+1,:]
        elif(len(col) ==2):
            tmp = arr[:,col[0]:col[1]+1:]
    else:
            # 1D
        if (type(row) == str):
            tmp = arr[:,col].reshape(-1,1)
        elif(type(col) == str):
            tmp = arr[row,:].reshape(1,-1)
        else:
            pass            #tmp = arr[row,col]   

    return tmp
#-----------------------------------------------------------------
def BallTreedensity(points,weights,bandwidths,*args):
    nargin = len(locals())-1  # -1 cos of args
    if(nargin<4):
        ttype = 0

  # if a single element is passed in , convert to m n matrix    
    try:
        nr,nc = points.shape
    except:
        if(points.size == 1):
            pts = points
            points = np.array([[pts]]) 
            
    try:
        nr,nc = weights.shape
    except:
        if(weights.size == 1):
            wts = weights
            weights = np.array([[wts]])
    
    Prob = BallTreeDens_module.BallTreeDens(points,weights,bandwidths,ttype)   
    dens = {}
    dens['type'] = np.array(Prob.get_type())
    dens['D'] = Prob.get_D()
    dens['N'] = Prob.get_N()
    dens['centers'] = Prob.get_centers()
    dens['ranges'] = Prob.get_ranges()
    dens['weights'] = Prob.get_weights().flatten()
    dens['lower'] = Prob.get_lower().flatten()
    dens['upper'] = Prob.get_upper().flatten()
    dens['leftch'] = Prob.get_leftch().flatten()
    dens['rightch'] = Prob.get_rightch().flatten()
    dens['perm'] = Prob.get_perm().flatten().flatten()
    dens['means'] = Prob.get_means()
    dens['bandwidth'] = Prob.get_bandwidth()    
    dens['class'] = "BTD"
    return dens
#-----------------------------------------------------------------

def knn(dens,points,k):
    temp1 = dens.copy()  # reference 
    N = dens['N']
    D = dens['D']
    ttype = dens['type']
    """
    bandwidth = dens['bandwidth']
    means = dens['means']
    perm = dens['perm']
    leftch = dens['leftch']
    rightch = dens['rightch']
    upper = dens['upper']
    lower = dens['lower']
    weights = dens['weights']
    ranges = dens['ranges']
    centers = dens['centers']
    
    knn_mod = knn_module.knn(D,N,ttype,centers,ranges,weights,lower,upper,leftch,
                             rightch,perm,means,bandwidth,points,k,points.shape[1])
    """
    # this had to be done cos C++ map is [string, eigen_mat]
    temp1['N'] = np.array((1,0))
    temp1['D'] = np.array((1,0))
    temp1['type'] = np.array((1,0))
    temp1['N'][0] = N
    temp1['D'][0] = D
    temp1['type'][0] = ttype
    del temp1['class']  # removes the class in dens
    knn_mod = knn_module.knn(temp1,points,k,points.shape[1])    
    distance = knn_mod.get_dist()
    neighbors = knn_mod.get_array()-1  
    return neighbors,distance

#-----------------------------------------------------------------
def BallTree(points,weights):
    nargin = len(locals())-1  # -1 cos of args
    if(nargin<4):
        ttype = 0
           
        
         
    Prob = BallTree_module.BallTree(points,weights)   # memory Leakage
    dens = {}
    dens['type'] = np.array(Prob.get_type())
    dens['D'] = Prob.get_D()
    dens['N'] = Prob.get_N()
    dens['centers'] = Prob.get_centers()
    dens['ranges'] = Prob.get_ranges()
    dens['weights'] = Prob.get_weights().flatten()
    dens['lower'] = Prob.get_lower().flatten()
    dens['upper'] = Prob.get_upper().flatten()
    dens['leftch'] = Prob.get_leftch().flatten()
    dens['rightch'] = Prob.get_rightch().flatten()
    dens['perm'] = Prob.get_perm().flatten().flatten()
    dens['means'] = Prob.get_means()
    dens['bandwidth'] = Prob.get_bandwidth()    
    dens['class'] = 'BT'
    return dens
#-----------------------------------------------------------------
def DualTree(dens1 = None,dens2=None,errTol=0.001):
    # note: Python dict is mutable 
    temp1 = dens1.copy()  # reference 
    
    
    N1 = temp1['N']
    D1 = temp1['D']
    ttype1  = temp1['type']
    temp1['N'] = np.array((1,0))
    temp1['D'] = np.array((1,0))
    temp1['type'] = np.array((1,0))
    
    temp1['N'][0] = N1
    temp1['D'][0] = D1
    temp1['type'][0] = ttype1
    cllass1 = temp1['class']
    del temp1['class']
    

    if(dens2 is None):      
         pp = DualTree_module.DualTree(temp1,errTol) 
         dens1['class'] = cllass1   
    else:
        temp2 = dens2.copy()
        N2 = temp2['N']
        D2 = temp2['D']
        ttype2  = temp2['type']
        temp2['N'] = np.array((1,0))
        temp2['D'] = np.array((1,0))
        temp2['type'] = np.array((1,0))
        
        temp2['N'][0] = N2
        temp2['D'][0] = D2
        temp2['type'][0] = ttype2
        cllass2 =temp2['class']
        del temp2['class']
        
        pp = DualTree_module.DualTree(temp1,temp2,errTol)       
        temp2['class'] = cllass2
        
    out = pp.get_plhs()
   
    return out
#-----------------------------------------------------------------
def rescale(npd,factor):
    factor = factor.reshape(-1,1)
    N = npd['N']
    npd['centers'] = npd['centers'] * np.tile(factor,(1,2*N)) 
    # reshape (-1,1) converts 1d array to row vector
     # reshape (1,-1) converts 1d array to column
    npd['ranges'] = npd['ranges'] * factor.max()
    npd['means'] = npd['means'] * np.tile(factor,(1,2*N))
    sz_bw = np.asarray(npd['bandwidth'].shape)
    sz_factor = np.asarray(factor.shape)  # shape returns tuple which is not operator overloaded
    npd['bandwidth'] = npd['bandwidth'] * np.tile(factor**2,(sz_bw/sz_factor).astype(int))
    return npd

#-----------------------------------------------------------------
def covar(dens,noBiasFlag =0):

    if(noBiasFlag != 0):
        cov,std_,mean_ = weighted_stats(getPoints(dens),getWeights(dens)) # code it
    else:
            if (dens['type'] == 0):
                cov = dens['bandwidth'][:,0] # Gaussian: store variances.
            elif(dens['type'] == 1):
                cov = 0.2* dens['bandwidth'][:,0]**2 # Epanetch BW -> variance
            elif(dens['type'] == 2):
                 cov = 0.2* dens['bandwidth'][:,0]**2 # Laplacian BW -> variance
                
    return cov
   
def weighted_stats(arr,wt):
    # funtion is a statsmodel wrapper
    weighted_stats = DescrStatsW(arr.T, weights=wt, ddof=0)
    # n weights must match the nrows of arr
    var = weighted_stats.var
    mean = weighted_stats.mean
    std =  weighted_stats.std
    return var,std,mean

#-----------------------------------------------------------------
def getBW(dens,ind,*args):
    nargin = len(locals())-1     
    if(nargin<2):
        ind = np.arange(0,dens['N'])
        
    ss = np.zeros((dens['D'],dens['N']))
    N = dens['N']
    tmp1 = dens['perm'][N + np.arange(N)] # debug - perm has dim of 1,D*N
    tmp2 = dens['bandwidth'][:,N + np.arange(N)]    
    ss[:,tmp1] = tmp2 ;
    ss = ss[:,ind]

    if (dens['type'] == 0) :
        ss = np.sqrt(ss) 
    out = np.zeros((dens['D'],1))
    out[:,0] = ss          
    return out
#-----------------------------------------------------------------
def getNpts(npd):
    N = npd['N'] ;
    return N
#-----------------------------------------------------------------
def getDim(npd):
    D = npd['D'] ;
    return D
#-----------------------------------------------------------------
def getNeff(npd):
    Neff = 1/sum(getWeights(dens)**2)
    return Neff
#-----------------------------------------------------------------
def getType(dens):    
    if(dens['type'] == 0) :
        typeS = 'Gaussian'
    elif(dens['type'] == 1):
        typeS = 'Epanetchnikov'
    elif(dens['type'] == 2):
        typeS = 'Laplacian'

    return typeS

#-----------------------------------------------------------------
def getPoints(dens,ind = None):
    
    if (ind is None):
           ind = np.arange(dens['N'])
       
    pts = np.zeros((dens['D'],dens['N']))
    tmp = dens['perm'][dens['N'] + np.arange(dens['N'])]  # generate 1D array
    pts[:,tmp]= dens['centers'][:,dens['N']+ np.arange(0,dens['N'])]
    pts = pts[:,ind]      
    return pts 
#-----------------------------------------------------------------
def getWeights(dens,*args):
    nargin = len(locals())-1 
    
    N = dens['N']
    wts = np.zeros((1,N))
    if (nargin <2):
        ind = np.arange(N)
    indx_L = dens['perm'][N + np.arange(N)]  # +1 compnent
    wts[dens['perm'][indx_L]] = dens['weights'][N + np.arange(N)]
    #wts = wts[:,ind]
    return wts[:,ind]
#-----------------------------------------------------------------
def double(npd):
    if(npd['N']>0):
        d=1
    else:
        d=0        
    
    return d

#-----------------------------------------------------------------

def entropy(x,ttype = "rs",arg1 = None):
    if (ttype == 'lvout'):
       varargin = {"lvout",arg1}
      

    p1 = 0 # undefined in the original code
    if (ttype == "rs" or ttype=="lln"):
        H = -evalAvgLogL(x,x)
        """
    elif(ttype,"rand"):
        N = len(varargin) # please checke
        ptsE = sample(p1,N)
        pE = kde(ptsE,1)
        KLD = evalAvgLogL(p1,pE) - evalAvgLogL(p2,pE)
        
    elif(ttype,"unscent"):          
        print("Not completed)
        D = getDim(x)
        N = getNpts(x)
        ptsE = getPoints(x)
        wts = getWeights(x)
        ptsE = np.tile(ptsE,(1,2*D+1)) # make 2*dim copies of each point
        wts =  np.tile(wts,(1,2*D+1) # (and its weight)
        bw = getBW(x,np.arange(N))
        """
    elif(ttype,"dist"):
        H=entropyDist(x)
    else:
        raise Exception ('Unknown entropy estimate method ''type''')
    return H

 #----------------------------------------------------------------- 
def marginal(dens,ind):
    pts = getPoints(dens)
    if (dens['bandwidth'].shape[1] > 2*dens['N']):
        sig = getBW(dens,np.arange(getNpts(dens)))
    else:
        sig = getBW(dens,1)
    
    wts = getWeights(dens)
    p = kde(extr_vec(pts,ind,'all'),extr_vec(sig,ind,'all'),wts,getType(dens))
    return p#-----------------------------------------------------------------
def nLOO_LL(alpha,npd,*args):
     nargin = len(locals())-1 
     if (nargin <2 ):
         raise Exception ('ksize: LOO_LL: Error!  Too few arguments')
     if(npd['type'] == 0):
         alpha = alpha**2
     npd['bandwidth'] = npd['bandwidth'] * alpha
     H = entropy(npd,'lvout')
     npd['bandwidth'] = npd['bandwidth'] / alpha 
     return H

#-----------------------------------------------------------------
def neighborDistance(npd,Nnear):
    nn,prop = knn(npd,getPoints(npd),round(Nnear)+1)
    minm, maxm = neighborMinMax(npd)
    return prop,minm,maxm

#-----------------------------------------------------------------
def neighborMinMax(npd):
    maxm = np.sqrt(np.sum((2*npd['ranges'][:,0])**2) );
    tmp = 2*npd['ranges'][:,np.arange(npd['N']-1)]
    tmp2 = np.sqrt(np.sum(tmp**2,axis = 0))  # sum along dimension 2
    minm = np.min(tmp2)
   # minm = np.minimum(np.sqrt(np.sum((2*npd['ranges'](:,np.arange(0,npd.N-1))**2 ,1)),[],2);
    minm = np.maximum(minm,1e-6);
    return minm,maxm

#-----------------------------------------------------------------
def nLSCV(alpha,npd):
     nargin = len(locals())-1 
     if(nargin <2):
         raise Exception('ksize: LSCV: Error!  Too few arguments')
     if(npd['type'] == 0):
        alpha = alpha**2
        
     npd['bandwidth'] =  npd['bandwidth']* 2*alpha
     H = means(evaluate(npd,npd))
     npd['bandwidth'] =  npd['bandwidth']/2
     H = H - mean(evaluate(npd,npd,'lvout'))
     npd['bandwidth'] =  npd['bandwidth']/alpha  
   
     return H

#-----------------------------------------------------------------
def ksizeLSCV(npd):
    minm,maxm = neighborMinMax(npd)
    npd = kde(getPoints(npd),(minm+maxm)/2,getWeights(npd),getType(npd))
    h =  golden(npd,nLSCV,2*minm/(minm+maxm),1,2*maxm/(minm+maxm),1e-2);
    h = h * (minm+maxm)/2
    return h
#-----------------------------------------------------------------   
def ksizeROT(npd,noIQR=0):

    X = getPoints(npd)
    N =X.shape[1]
    dim = X.shape[0]
    
    Rg = 0.282095
    Mg=1
    Re = 0.6
    Me = .199994
    Rl = 0.25
    Ml = 1.994473
    
    if(npd['type'] ==0):
        prop = 1
    elif(npd['type'] ==1):
        prop = ((Re/Rg)**dim / (Me/Mg)**2 )**(1/(dim+4))
    elif(npd['type'] ==2):
       prop = ((Rl/Rg)**dim / (Ml/Mg)**2 )**(1/(dim+4)) 
    
    sig = std(X,1)
    
    if(noIQR > 1):
      h = prop*sig*N^(-1/(4+dim))
    else:
     iqrSig = 0.7413*iqr(X)  # find interquartile range sigma est
     if(np.max(iqrSig) ==0):
        iqrSig=sig
    h = prop * np.minimum(sig,iqrSig) * N**(-1/(4+dim))

    return h

#-----------------------------------------------------------------
def iqr(x):
    xS = x.copy()
    xS = np.sort(xS.T,axis=0)
   # xS = xS[::-1] # reverse the order
  #  xS.sort()
    N = x.shape[1]
    out =  xS[math.ceil(3*N/4)-1,:] - xS[math.ceil(N/4)-1,:]
    return out
#-----------------------------------------------------------------
def entropyDist(npd):
    Ce = .57721566490153286
    pts = getPoints(npd)

    N1,N2 = pts.shape
    tmp,D = knn(npd,pts,2)
    Sr = N1* math.pi**(N1/2) / gamma((N1/2) + 1)
    h = N1/N2 * np.sum(np.log(D)) + np.log(Sr * (N2-1)/N1 ) + Ce
    return h

#-----------------------------------------------------------------   
def evalAvgLogL(dens,at,*args) :
    if(at['class'] =='kde'):
        L = evaluate(dens,at)
        W = getWeights(at)
        ind = np.nonzero(L==0)
        if(any(W[ind]) == True):
            ll = -math.inf
        else:
            L[ind] = 1
            ll = np.matmul(np.log(L),W.T)
    else:
        L = evaluate(dens,at)
        ind = np.nonzero(L==0) 
        if( len(ind)>1):
           ll = math.inf
        else:
           ll=np.mean(np.log(L))
    return ll
  
#-----------------------------------------------------------------
def evaluate(dens,pos,lvFlag = 0,errTol = 1e-3):
    if (type(pos) == dict):
        if (pos['class'] == 'kde'):
            posKDE = pos
            dim = getDim(pos)
    else:
        posKDE = BallTree(pos,np.ones((1,pos.shape[1]))/pos.shape[1])
        dim = pos.shape[0]
    if(getDim(dens) !=dim):
        raise Exception('X and Y must have the same dimension')
    if( lvFlag ==1):
        pp = DualTree(dens,errTol)
    else:
        pp = DualTree(dens,posKDE,errTol) 
 
    return pp
#-----------------------------------------------------------------
def sample(npd,Npts,*args):
   nargin = len(locals())-1 
   if (nargin <3):
      points = np.zeros((getDim(npd),Npts))
      ind = np.zeros(1,Npts)
   
   bw = getBW(npd)
   w = getWeights(npd)
   w = w.cumsum()
   w = w/w[-1]
   return points,ind

#-----------------------------------------------------------------
def randkernel(N,M,ttype):
    ttype = ttype.lower()
    ttype = ttype[0]
    
    if (ttype == 'g'):
        samples = randNormal(N,M)
    elif(ttype == 'l'):
        samples = randLaplace(N,M)
    elif(ttype == 'e'):
        samples = randEpanetch(N,M)   
        
    return samples

#----------------------------------------------------------------- 
def randNormal(N,M):
    samples = np.random.normal(loc=0,scale=1,size=(N,M))
    return samples

#----------------------------------------------------------------- 
def randEpanetch(N,M):
    ii = 0 + 1j
    u= np.random.rand(loc=0,scale=1,size=(N,M))
    a2 = 0
    a1=-3
    a0=4*u-2
    Q = 1/3 * a1;                     
    R = 1/2 * (-a0);
    D = Q**3 + R**2;
    S = (R + np.sqrt(D))**(1/3)
    T = (R - np.sqrt(D))**(1/3)
    ans3 = -.5*(S+T) - .5*np.sqrt(3)*ii*(S-T)
    samples=np.zeros((N,M))
    F = abs()
    
    tmp = samples[F]
    if(tmp.nonzero()):
        print("!")
    samples[F] = ans3[F]    
    return samples

#----------------------------------------------------------------- 
def randLaplace(N,M):
    binary = np.random.rand(loc=0,scale=1,size=(N,M))
    binary = np.nonzero(binary>0.5)
    binary = 2*binary -1
    samples = binary * math.log(np.random.rand(N,M))
    return samples

#-----------------------------------------------------------------
def mean(dens):
    return dens['means'][:,0]

#-----------------------------------------------------------------
def maxx(dens):
    L = evaluate(dens,getPoints(dens))
    #mx = np.max(L)
    mxind = np.argmax(L)
    x = getPoints(dens,mxind) # x = getPoints(dens,mxind[0])
    return x

#-----------------------------------------------------------------
def golden(npd,func, ax, bx, cx, tol,*args):
    C = (3-math.sqrt(5))/2
    R = 1-C
    x0 = ax
    x3 = cx
    
    if(abs(cx-bx) > abs(bx-ax)):
        x1=bx
        x2 = bx + C*(cx-bx)
    else:
        x2 = bx
        x1 = C*(bx-ax)
    
    f1 = func(x1,npd)  # passing in func as function handle
    f2 = func(x2,npd)
    k=1
    
    while abs(x3-x0) > tol*(abs(x1)+abs(x2)):
      if (f2 < f1):
        x0 = x1;
        x1 = x2;
        x2 = R*x1 + C*x3;   
        f1 = f2;
        f2 = func(x2,npd);
      else:
        x3 = x2;
        x2 = x1;
        x1 = R*x2 + C*x0;   
        f2 = f1;
        f1 = func(x1,npd)

      k = k+1
    if (f1 < f2):
      xmin = x1;
      fmin = f1;
    else:
      xmin = x2;
      fmin = f2;

    return xmin,fmin

#-----------------------------------------------------------------
    
def kde(points,ks = None,weights = None,typeS = None,*args): 
    
    if(ks is None):
        nargin = 2
    elif(weights is None):
        nargin = 3
    elif(typeS is None):
        nargin = 4
    else:
        nargin = 5
    
    ttype= 0  # type is an in-built python Function
    kstype = None

    if (nargin ==1 and points['class'] == kde ):
        # CONSTRUCTOR FROM KDE-TYPE
        ttype = points['type']
        weights = points['weights']
        
        if(points['bandwidth'].shape[1]> 2*points['N']) :    
            ks = getBW(points,np.arange(getNpts(points)))
        else:
            ks = getBW(points,1)            
            if (ttype == 0) :
                ks = ks**2        
            points = getPoints(points)
    elif(nargin ==1): # ugh, needed for deserialization
        npts = points['N']*points['D']
        p = points['centers']
        print("I m side kde function -- when deserializing")       
    

# CONSTRUCTOR FROM RAW DATA
    elif (nargin> 0 ): # construct from raw data     
  #      raise Exception ('Pass sumtin nin')
        
        if (type(ks) == str):
            kstype = ks
            ks = np.array([1])
        if (ks.shape[0] ==1):
            ks = np.tile(ks,(m_size(points)[0],1)) # ks = repmat(ks,[size(points,1),1])
        if(nargin<3):
            weights = np.ones((1,points.shape[1]))  # ones(1,size(points,2))
        if (weights is None) :  # equivalent of Matlab exist or empty
            weights = np.ones((1, points.shape[1]))  # shape[0] - rows shape[1] - columns
        if(nargin <4):
            typeS = 'g'
        else:
            typeS = typeS.lower();
            typeS = typeS[0]
            
        if (typeS == 'l') :
            ttype = 2
        elif (typeS == 'e') :
            ttype = 1
        elif (typeS == 'g') :
            ttype = 0
            ks = ks**2
        else:
            print ('Type must be one of (G)aussian, (L)aplacian, or (E)panetchnikov')
                  
        if(type(weights) == tuple):
            pass
        else:
            weights = weights/np.sum(weights)
        
      # Check matrix sizes
        D,N = m_size(points)
    
        if (m_size(weights) != (1, N)):
          raise Exception ('Weights must be [1xNpoints] (or empty)')     
        bwsize = ks.shape
    
        if (bwsize !=  (1,1) and bwsize != (D,1) and bwsize != (D,N) and bwsize != (D,)): # kszie changes ks
          raise Exception ('Bandwidth must be scalar, [Dx1], [DxN], or an automatic selection method')
             
    else:
        #  EMPTY CONSTRUCTOR
        points = []
        ks = []
        weights = []             
             
    prob = BallTreedensity(points,weights,ks,ttype)
    prob['class'] = 'kde'
    if (kstype is not None): 
      prob = ksize(prob,kstype)    

    return prob
#-----------------------------------------------------------------
def ksize(npd,ttype,*args):
    nargin = len(locals())-1
    Nd = npd['D']
    Np = npd['N']
    
    if (Np ==1):
        npd =kde(getPoints(npd),0,getWeights(npd),getType(npd))
        
    if (nargin <2):
        ttype = 'lcv'
        
    if(type(ttype) == "double"):
        ks = ttype
        if(len(ks)==1):
            ks = ks+ np.zeros(Nd,1)
        npd = kde(getPoints(npd),ttype,getWeights(npd),getType(npd))
        return
    
    ttype = ttype.lower()
    if (ttype[-1] == 'p'):
        stddv = covar(npd,0)          #pre-equalize variances for 
        npd = rescale(npd, 1/stddv)   #any 1-d calculations
        
#.... KERNEL SIZE SELECTION METHODS...... 
#  least-squares cross-validation
    elif(ttype.lower() == 'lscv' or ttype.lower()  =='lscvp'): 
        ks = ksizeLSCV(npd)
#   (uniform) likelihood cross-validation       
    elif (ttype.lower() == 'lcv'or ttype.lower() =='unif' or ttype.lower()=='lcvp'or ttype.lower()=='unifp'):
        minm,maxm = neighborMinMax(npd)
        npd = kde(getPoints(npd),(minm+maxm)/2,getWeights(npd),getType(npd))
        ks =  golden(npd,nLOO_LL,2*minm/(minm+maxm),1,2*maxm/(minm+maxm),1e-2)
        ks = ks * (minm+maxm)/2; 
#   local likelihood cross-val      
    elif(ttype.lower()=='local' or ttype.lower()=='localp'):
         if (nargin < 3):
            [prop,minm,maxm] = neighborDistance(npd,np.sqrt(getNpts(npd)));
         else:
             prop = varargin[0]
             minm,maxm = neighborMinMax(npd)
         prop = prop / mean(prop)
         npd = kde(getPoints(npd),(minm+maxm)/2 * prop,getWeights(npd),getType(npd))
         ks =  golden(npd,nLOO_LL,2*minm/(minm+maxm),1,2*maxm/(minm+maxm),1e-2)
         ks = ks * (minm+maxm)/2 * prop
    
#    STANDARD-DEVIATION BASED %%%% 
    elif(ttype.lower()== 'rot'):
       ks = ksizeROT(npd);  # % "Rule of Thumb" (stddev-based)
    elif(ttype.lower() == 'msp'): # Maximal Smoothing Principle"
        ks = ksizeMSP(npd)   # same as ROT but different constants
    elif(ttype.lower()== 'hall'or ttype.lower()=='hsjm'):
        ks = ksizeHall(npd)
    elif(ttype.lower()=='maxmin'):
        nn,prop = knn(npd,getPoints(npd),1+1);  
    
    if(ttype[-1] == 'p'):
        ks = np.tile(ks,[Nd,1]/[ks.shape[0],1]);
        ks = ks * np.tile(stddv,size.shape/stddv.shape) # fix up prev. equalization
        npd = rescale(npd, stddv)
    
    npd = kde(getPoints(npd),ks,getWeights(npd),getType(npd))        

    return npd

#-----------------------------------------------------------------
def condition(dens,ind,A):
    if(dens['bandwidth'].shape[1] > 2*dens['N']):
        wNew1 = np.zeros(1,getNpts(dens))
        for i in range(0,getNpts(dens)):
            ktmp = kde(getPoints(dens,i),getBW(dens,i),1,getType(dens))
            wNew1[i] = evaluate(marginal(ktmp,ind),A[ind],0)
    else:
        bw = getBW(dens,0)
        if (m_size(A)[1] >1): # temp fix 
            wNew1 = evaluate( kde(A[ind,0],bw[ind],1,getType(dens)) ,marginal(dens,ind),0)
        else:
            if (A.size == 1):
                wNew1 = evaluate( kde(A,bw[ind],1,getType(dens)) ,marginal(dens,ind),0)
            else:
                wNew1 = evaluate( kde(A[ind,0],bw[ind],1,getType(dens)) ,marginal(dens,ind),0)
    
    wNew = wNew1 * getWeights(dens)
    pts = getPoints(dens) 
    
    if(dens['bandwidth'].shape[1] > 2*dens['N']):
        bw = getBW(dens,np.aray(1,getNpts(dens)))
    else:
        bw = getBW(dens,1)
    newInd = np.setdiff1d(np.arange(getDim(dens)),ind)
    pp = kde(pts[newInd,:],bw[newInd,:],wNew,getType(dens))        
    return pp

#*****************************************************************************
if __name__ == '__main__':
    # Testing definitions
    #data = np.array([[1, 0.85, 0.65, .12],[0.5 ,0.6 ,1.2 ,0.5],[500,1200,700,5000]])
    data = np.array([[1, 0.85, 0.65, .12],[0.5 ,0.6 ,1.2 ,0.5]])

    weights = np.array([0.25,0.25,0.25,0.25])
    ks = np.array([1, 1])
    ttype = 0

    bw = np.array([[0.2039,0.2867, 0.1031,0,0.0931,0.0931,0.0931,0.0931],
                   [0.088,0.0035,0.0935,0,0.0035,0.0035,0.0035,0.0035]])
    ts = bw[0,:]
    sz = m_size(ts)

    dens = BallTreedensity(data,weights,ks)

    #get_point_dens = np.array([[1,0.85,0.65,0.12],[0.5,0.6,1.2,0.5]])
    #xx,yy = knn(dens,get_point_dens,2)
    #dens_rescale = rescale(dens,10) ;
    dens['bandwidth'] = bw 
    pt = getPoints(dens)

    #xx = getBW(dens) 
    wts = getWeights(dens)
    cov = covar(dens,1)
    npd = rescale(dens,1/cov) 

    dens2 = dens.copy
    p = DualTree(dens,dens)
    #prop,minm,maxm = neighborDistance(dens,2) # knn
    #dens = BallTreedensity(data,weights,ks)


    #posKDE = BallTree(data,weights)
    #p = DualTree(dens)


    """
    minm,maxm = neighborMinMax(dens)
    #prop,minm,maxm = neighborDistance(dens,2)

    #ks = golden(dens,nLOO_LL,2*minm/(minm+maxm),1,2*maxm/(minm+maxm),1e-2)

    #-----------------------------------
    # testing balltree

    posKDE = BallTree(data,weights)
    #-----------------------------------

    # testing Dualtree
    #-----------------------------------

    #p = DualTree(dens,posKDE)

    #-----------------------------------
    # testing entropy
    #-----------------------------------
    dens['class'] = 'kde'
    #p = entropy(dens)
    # testing entropyDist
    #-----------------------------------
    #dens['class']='kde'
    #p = entropy(dens,'dist')

    #-------------------kde-----------
    prob = kde(data,'rot')
    #xd  =getBW(prob,0)
    #-------------------marginal-----------
    #prob_marginal = marginal(prob,0)
    """
    """
    #-------------------condition-----------
    data = np.array([[1, 0.85, 0.65, .12],[0.5 ,0.6 ,1.2 ,0.5]])
    prob = kde(data,'rot')
    Prob_cond = condition(prob,0,extr_vec(data,0,'all'))
    mean(Prob_cond)
    maxx(Prob_cond) 




    # Multidimensional----------------------------
    data = np.array([[1, 0.85, 0.65, .12],[0.5 ,0.6 ,1.2 ,0.5],[500,1200,900,6000]])
    prob = kde(data,'rot')
    Prob_cond = condition(prob,np.array([0, 1]),extr_vec(data,[0,1],'all'))
    Prob_cond = condition(prob,np.array([0, 1]).T,data[0:2,:])
    mean(Prob_cond)
    maxx(Prob_cond) 
    """
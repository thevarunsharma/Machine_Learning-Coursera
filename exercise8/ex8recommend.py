import numpy as np
from scipy.io import loadmat
import scipy.optimize as spo

data=loadmat('ex8_movies.mat')
r=data['R']
y=data['Y']
nm,nu=y.shape
param_data=loadmat('ex8_movieParams.mat')
theta=param_data['Theta']
X=param_data['X']
params=np.concatenate((X.ravel(),theta.ravel()))

def cost_reg(params,y,r,n,l):
    X=np.reshape(params[:nm*n],(nm,n))
    theta=np.reshape(params[nm*n:],(nu,n))
    err=(X.dot(theta.T)-y)*r
    J=np.sum(err**2.0)/2.0+l*(np.sum(theta**2.0)+np.sum(X**2.0))/2
    return J
"""
def grad(params,y,r,n,l):
    X=np.reshape(params[:nm*n],(nm,n))
    theta=np.reshape(params[nm*n:],(nu,n))
    err=(X.dot(theta.T)-y)*r
    dJX=err.dot(theta)+l*X
    dJtheta=err.T.dot(X)+l*theta
    dJ=np.concatenate((dJX.ravel(),dJtheta.ravel()))
    return dJ
"""
lr=10
org_params=spo.fmin_bfgs(f=cost_reg,x0=params,args=(y,r,10,lr))
f=open('params.txt','w+')
f.write(str(org_params))
f.close()

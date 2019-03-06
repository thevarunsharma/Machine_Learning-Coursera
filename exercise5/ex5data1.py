import numpy as np
import scipy.io as spio
import scipy.optimize as spo

data=spio.loadmat("ex5data1.mat")
"""
Xval=data["Xval"].T[0]
yval=data['yval'].T[0]
Xval=np.array([np.ones(Xval.size),Xval])
l=0
"""
theta=np.zeros(9)
def hypo(theta,X):
    return theta.dot(X)

def costJ(theta,X,y):
    m=float(y.size)
    err=hypo(theta,X)-y
    J=np.sum(err**2.0)/(2*m)
    return J

def gradJ(theta,X,y):
    m=float(y.size)
    err=hypo(theta,X)-y
    dJ=err.dot(X.T)
    return dJ
X1=(data["X"].T)[0]
y=(data["y"].T)[0]
X=np.array([np.ones(X1.size),X1,X1**2.0,X1**3.0,X1**4.0,X1**5.0,X1**6.0,X1**7.0,X1**8.0])
optimum_t=spo.fmin(func=costJ,x0=theta,args=(X,y))
print optimum_t

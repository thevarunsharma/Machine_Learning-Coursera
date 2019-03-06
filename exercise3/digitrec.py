import numpy as np
import scipy.optimize as spo
import scipy.io as spio

data=spio.loadmat("ex3data1.mat")
X=data['X']*10
X=np.concatenate((np.ones((5000,1)),X),axis=1)
X=X.T
y=data['y']
y=y%10
y=y.T
l=1
def ht(theta,X):
    sig=1/(1+np.exp(-theta.dot(X)))
    return sig

def costJ(theta,X,y):
    m=float(y.size)
    hypo=ht(theta,X)
    cost=-sum(y*np.log(hypo)+(1-y)*np.log(1-hypo))/m+sum(theta[1:]**2.0)*l/(2*m)
    return cost

def dJ(theta,X,y):
    m=float(y.size)
    hypo=ht(theta,X)
    err=hypo-y
    d=(err.dot(X.T))/float(m)-l*theta/m
    return d

theta=np.zeros((10,401))
for k in xrange(10):
    yi=np.zeros(5000)
    yi[500*k:500*(k+1)]=1
    t=np.zeros(401)
    theta[k]=spo.fmin_cg(f=costJ,x0=t,fprime=dJ,args=(X,yi))
print theta
print theta.shape

def predict(Xi):
    hypo=1/(1+np.exp(-theta.dot(Xi.T)))
    return hypo.argmax()            #argmax return index of maximum element

import numpy as np
import scipy.io as spio
import scipy.optimize as spo

theta=spio.loadmat("ex3weights.mat")
theta1=theta["Theta1"]
theta2=theta["Theta2"]
data=spio.loadmat("ex3data1.mat")
X=data['X']
X=np.concatenate((np.ones((5000,1)),X),axis=1)
X=X.T
y=data['y']

def predict(Xi,theta):
    hypo=1/(1+np.exp(-theta.dot(Xi)))
    return hypo
a1=predict(X,theta1)
print a1
print a1.shape
a1=np.concatenate(([np.ones(5000)],a1),axis=0)
res=predict(a1,theta2)
print res.shape
res=res.argmax(axis=0)+1

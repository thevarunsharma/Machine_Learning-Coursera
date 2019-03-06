import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats

def estimate_gaussian(X):
    mu=X.mean(axis=0)
    sigma2=X.var(axis=0)
    return mu,sigma2

def select_threshold(pval,yval):
    bestE=0
    bestF1=0
    step=(pval.max()-pval.min())/1000
    for E in np.arange(pval.min(),pval.max(),step):
        pred=(pval.prod(axis=1)<E).astype(int)    #anomaly
        tpos=np.sum(np.logical_and(pred==1,yval==1).astype(float))
        fpos=np.sum(np.logical_and(pred==1,yval==0).astype(float))
        fneg=np.sum(np.logical_and(pred==0,yval==1).astype(float))
        pre=tpos/(tpos+fpos)
        rec=tpos/(tpos+fneg)
        F1=2*pre*rec/(pre+rec)
        if F1>bestF1:
            bestF1=F1
            bestE=E
    return bestE,bestF1

data2=loadmat('ex8data2.mat')
X=data2['X']
Xval=data2['Xval']
yval=data2['yval'].T[0]
mu,var=estimate_gaussian(X)
pval=np.zeros(Xval.shape)
for i in range(X.shape[1]):
    pval[:,i]=stats.norm(mu[i],var[i]).pdf(Xval[:,i])

E,F1=select_threshold(pval,yval)
print E,F1

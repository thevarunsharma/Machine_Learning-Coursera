import numpy as np
from sklearn import svm
from scipy.io import loadmat

gamma_values=[0.01,0.03,0.1,0.3,1,3,10,30,100]
C_values=[0.01,0.03,0.1,0.3,1,3,10,30,100]
data=loadmat("ex6data3.mat")
X=data['X']
y=data['y'].T[0]
Xval=data['Xval']
yval=data['yval']

best_score=0
best_params={'C':None,'gamma':None}

for C in C_values:
    for gamma in gamma_values:
        svc=svm.SVC(C=C,gamma=gamma)
        svc.fit(X,y)
        score=svc.score(Xval,yval)

        if score>best_score:
            best_score=score
            best_params['C']=C
            best_params['gamma']=gamma

print best_score
print best_params

import numpy as np
import scipy.io as spio
from sklearn.preprocessing import OneHotEncoder

data=spio.loadmat("ex4data1.mat")
X=data['X']
y=data['y']%10
encode=OneHotEncoder(sparse=False)
y_hot=encode.fit_transform(y)
input_size=400
hidden_size=25
num_labels=10
learning_rate=1
params=np.random.random(hidden_size*(input_size+1)+num_labels*(hidden_size+1))*0.5-0.25

def sigmoid(z):
    return 1/(1+np.exp(-z))

def ForwardProp(X,theta1,theta2):
    m=X.shape[0]
    a1=np.insert(X,0,values=np.ones(m),axis=1)
    z2=a1.dot(theta1.T)
    a2=np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
    z3=a2.dot(theta2.T)
    hypo=sigmoid(z3)
    return a1,z2,a2,z3,hypo
"""
def cost(params,input_size,hidden_size,num_labels,X,y,learning_rate):
    m=X.shape[0]
    #reshaping parameter vector
    theta1=np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1)))
    theta2=np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1)))
    #running ForwardProp
    a1,z2,a2,z3,h=ForwardProp(X,theta1,theta2)
    #compute cost
    J=0
    for i in xrange(m):
        first=-y[i]*np.log(h[i])
        second=-(1-y[i])*np.log(1-h[i])
        J+=np.sum(first+second)
    J=J/m
    J+=(float(learning_rate)/(2*m))*(np.sum(theta1[:,1:]**2)+np.sum(theta2[:,1:]**2))

    return J
"""
def sigmoid_gradient(z):
    return np.multiply(sigmoid(z),(1-sigmoid(z)))

def BackProp(params,input_size,hidden_size,num_labels,X,y,learning_rate):
    m=X.shape[0]
    X=np.matrix(X)
    y=np.matrix(y)
    #reshaping parameter vector
    theta1=np.matrix(np.reshape(params[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
    theta2=np.matrix(np.reshape(params[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))
    #running ForwardProp
    a1,z2,a2,z3,h=ForwardProp(X,theta1,theta2)

    J=0
    delta1=np.zeros(theta1.shape)
    delta2=np.zeros(theta2.shape)
    #compute cost
    for i in xrange(m):
        first=-np.multiply(y[i],np.log(h[i]))
        second=-np.multiply((1-y[i]),np.log(1-h[i]))
        J+=np.sum(first+second)
    J=J/m
    #add cost regularisation term
    J+=(float(learning_rate)/(2*m))*(np.sum(np.power(theta1[:,1:],2))+np.sum(np.power(theta2[:,1:],2)))
    #main back BackProp
    for t in xrange(m):
        a1t=a1[t,:]
        z2t=z2[t,:]
        a2t=a2[t,:]
        ht=h[t,:]
        yt=y[t,:]

        d3t=ht-yt

        z2t=np.insert(z2t,0,values=np.ones(1))
        d2t=np.multiply((theta2.T).dot(d3t.T).T,sigmoid_gradient(z2t))

        delta1=delta1+d2t[:,1:].T.dot(a1t)
        delta2=delta2+d3t.T.dot(a2t)
    delta1=delta1/m
    delta2=delta2/m
    #add gradient regularisation term
    delta1[:,1:]=delta1[:,1:]+theta1[:,1:]*learning_rate/m
    delta2[:,1:]=delta2[:,1:]+theta2[:,1:]*learning_rate/m
    #unroll grad matrix to vector
    grad=np.concatenate((delta1.ravel(),delta2.ravel()),axis=1)

    return J,grad

from scipy.optimize import minimize
fmin=minimize(fun=BackProp,x0=params,args=(input_size,hidden_size,num_labels,X,y_hot,learning_rate),method='TNC',jac=True,options={'maxiter':250})
classifier=map(float,fmin.x)
fh=open("classifier.txt","w+")
fh.write(str(classifier))
fh.close()
X=np.matrix(X)
t1=np.matrix(np.reshape(fmin.x[:hidden_size*(input_size+1)],(hidden_size,(input_size+1))))
t2=np.matrix(np.reshape(fmin.x[hidden_size*(input_size+1):],(num_labels,(hidden_size+1))))

a1,z2,a2,z3,h=ForwardProp(X,t1,t2)
y_pred=np.array(np.argmax(h,axis=1))
correct=[1 if a==b else 0 for (a,b) in zip(y_pred,y)]
acc=sum(correct)/float(len(correct))
print "accuracy=",acc*100,"%"
